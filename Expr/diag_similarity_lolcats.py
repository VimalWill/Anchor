"""diag_similarity.py for LoLCATs: compare attention readouts across layers.

A LoLCATs layer mixes softmax over the last `window_size` keys with linear
attention over everything older, under one shared normalizer. The output is
split exactly along that line, in residual-stream coordinates (o_proj is
bias-free, so the parts sum to the layer output):

  state  -- the linear-attention (recurrent state) contribution; the analogue
            of the TTT history effect, since the state starts at zero
  window -- the sliding-window softmax contribution
  total  -- state + window, checked against the layer's actual output

Metrics, aggregation and plots are shared with diag_similarity.py.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from diag_similarity import aggregate_pairs, pairwise_similarity, plot_similarity, query_positions
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root: eval, Training, LinearTTT

COMPONENTS = ('state', 'window', 'total')


@torch.no_grad()
def split_readout(attn, hidden_states, position_ids, positions):
    """Recompute hybrid_attention_quadratic for the sampled query rows only."""
    q, k, v, _ = attn._process_qkv(hidden_states, position_ids=position_ids)
    f_q, f_k = attn.feature_map_q(q), attn.feature_map_k(k)
    idx = positions.to(q.device)
    q, f_q = q[:, :, idx].float(), f_q[:, :, idx].float()
    keys = torch.arange(k.shape[-2], device=q.device)
    offset = idx[:, None] - keys[None, :]  # query minus key position
    in_window = (offset >= 0) & (offset < attn.window_size)
    in_linear = offset >= attn.window_size

    window_factor = torch.sigmoid(attn.window_factors.float())
    linear_factor = 1 - window_factor if attn.affine_attention_factors else 1
    a_sm = torch.einsum('bhmd,bhnd->bhmn', q, k.float()) * (k.shape[-1] ** -0.5)
    a_sm = a_sm.masked_fill(~in_window, -1e8)
    a_sm = window_factor * torch.exp(a_sm - a_sm.amax(-1, keepdim=True))
    a_ln = linear_factor * torch.einsum('bhmd,bhnd->bhmn', f_q, f_k.float()).masked_fill(~in_linear, 0)
    denominator = a_sm.sum(-1, keepdim=True) + a_ln.sum(-1, keepdim=True) + 1e-6

    def project(weights):
        y = (torch.einsum('bhmn,bhnd->bhmd', weights, v.float()) / denominator).to(q.dtype)
        y = y.to(hidden_states.dtype).transpose(1, 2).reshape(len(idx), attn.hidden_size)
        return attn.o_proj(y).float()

    state, window = project(a_ln), project(a_sm)
    return {'state': state.cpu(), 'window': window.cpu(), 'total': (state + window).cpu()}


@torch.no_grad()
def collect_readouts(model, layers, ids, positions):
    """Read-only hooks. Returns layer -> component -> [sampled queries, hidden]."""
    hooks, inputs, result = [], {}, {}

    def capture(layer_id):
        def hook(module, args, kwargs):
            hidden = kwargs['hidden_states'] if 'hidden_states' in kwargs else args[0]
            inputs[layer_id] = (hidden, kwargs.get('position_ids'))
        return hook

    def finish(layer_id):
        def hook(module, args, kwargs, output):
            hidden, position_ids = inputs.pop(layer_id)
            parts = split_readout(module, hidden, position_ids, positions)
            actual = output[0][0].index_select(0, positions.to(output[0].device)).float().cpu()
            error = (parts['total'] - actual).norm() / actual.norm().clamp_min(1e-12)
            if error > 2e-2:  # bf16 projections; a real mismatch is O(1)
                raise RuntimeError(f'Layer {layer_id}: split readout differs from the '
                                   f'layer output (relative error {error:.3g})')
            result[layer_id] = parts
        return hook

    try:
        for layer_id in layers:
            attn = model.model.layers[layer_id].self_attn
            hooks.append(attn.register_forward_pre_hook(capture(layer_id), with_kwargs=True))
            hooks.append(attn.register_forward_hook(finish(layer_id), with_kwargs=True))
        model(input_ids=ids.to(next(model.parameters()).device), use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    missing = set(layers) - set(result)
    if missing:
        raise RuntimeError(f'Readout hooks did not run at layers {sorted(missing)}')
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ckpt', required=True, help='stage-1 LoLCATs model')
    ap.add_argument('--adapter', help='stage-2 LoRA adapter, merged before probing')
    ap.add_argument('--tokens-file', help='reuse exact corpus tokens from previous probes')
    ap.add_argument('--corpora', nargs='+', choices=['pg19', 'wikitext'], default=['pg19', 'wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192); ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--samples', type=int, default=128, help='max query positions per slice per sequence')
    ap.add_argument('--slices', nargs='+', choices=['cross_window', 'other'], default=['cross_window', 'other'])
    ap.add_argument('--slice-window', type=int,
                    help="recall distance that defines cross_window (default: the model's window)")
    ap.add_argument('--layers', nargs='+', type=int, help='default: all layers')
    ap.add_argument('--seed', type=int, default=42); ap.add_argument('--shuffles', type=int, default=5)
    ap.add_argument('--analysis-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--save-features', action='store_true'); ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if args.samples < 4 or args.seqs < 1 or args.seq_len < 5 or args.shuffles < 1:
        ap.error('Need samples >= 4, positive sequence count, length >= 5, and shuffles >= 1')
    if not torch.cuda.is_available():
        ap.error('Requires CUDA')

    root = Path(__file__).resolve().parents[1]
    sys.path.append(str(root / 'third_party' / 'Liger'))  # appended: Liger's eval/ must not shadow eval.py
    import fla  # noqa: F401
    import lolcats  # noqa: F401  registers the model with AutoModelForCausalLM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from eval import ar_masks
    from diag_rank import checkpoint_files, load_corpora, validate_tokens

    tokenizer = AutoTokenizer.from_pretrained(args.ckpt)
    model = AutoModelForCausalLM.from_pretrained(args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda')
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    model.eval().requires_grad_(False)
    window = model.model.layers[0].self_attn.window_size
    slice_window = args.slice_window or window
    layers = args.layers if args.layers is not None else list(range(len(model.model.layers)))
    if set(layers) - set(range(len(model.model.layers))):
        raise ValueError('Requested layer is not present in model')

    args.base = tokenizer.name_or_path  # what load_corpora and token files key on
    if args.tokens_file:
        saved = torch.load(args.tokens_file, map_location='cpu', weights_only=True)
        if saved['tokenizer'] != args.base:
            print(f"warning: token file was made with {saved['tokenizer']}, model uses {args.base}")
        corpora = {name: saved['corpora'][name] for name in dict.fromkeys(args.corpora)}
        sources = {name: saved['sources'][name] for name in corpora}
    else:
        corpora, sources = load_corpora(args, tokenizer)
    validate_tokens(corpora, args.seq_len, args.seqs, model.config.vocab_size)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=False)
    torch.save({'tokenizer': args.base, 'corpora': corpora, 'sources': sources}, out / 'tokens.pt')
    identity = {name: checkpoint_files(path) for name, path in (('ckpt', args.ckpt), ('adapter', args.adapter))}
    metadata = {'complete': False, 'arguments': vars(args), 'sources': sources,
                'model_config': model.config.to_dict(), 'checkpoint_files': identity,
                'window_size': window, 'slice_window': slice_window, 'layers': layers,
                'token_hashes': {name: hashlib.sha256(ids.numpy().tobytes()).hexdigest() for name, ids in corpora.items()},
                'definitions': {'state': 'linear-attention contribution to the layer output (keys older than the window)',
                    'window': 'sliding-window softmax contribution to the layer output',
                    'total': 'state + window = the attention layer output',
                    'cross_window': f'repeated-bigram targets with source more than {slice_window} tokens back',
                    'other': f'non-repeated-bigram targets at query positions >= {window}'},
                'similarity_reference': 'https://proceedings.mlr.press/v97/kornblith19a.html',
                'limitations': 'Similarity is not mutual information, state identity, or proof of causal redundancy.',
                'torch_version': str(torch.__version__)}
    metadata['code_hashes'] = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                               for name in ('Expr/diag_similarity_lolcats.py', 'Expr/diag_similarity.py', 'Expr/diag_rank.py', 'eval.py',
                                            'third_party/Liger/lolcats/models/lolcats/modeling_lolcats.py')}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')

    rows, selections, magnitudes = [], [], []
    names = {'cross_window': f'ar_{slice_window}+', 'other': 'other'}
    for corpus, sequences in corpora.items():
        for sequence, ids in enumerate(sequences):
            masks = ar_masks(ids, [slice_window, float('inf')])
            # min position = window: earlier queries have an empty state
            chosen = {name: query_positions(masks[names[name]], window, args.samples, args.seed + sequence)
                      for name in dict.fromkeys(args.slices)}
            selections.append({'corpus': corpus, 'sequence': sequence,
                               'positions': {k: v.tolist() for k, v in chosen.items()}})
            chosen = {k: v for k, v in chosen.items() if len(v) >= 4}
            if not chosen:
                print(f'{corpus} {sequence + 1}: no slice with four queries, skipping', flush=True)
                continue
            positions = torch.cat(list(chosen.values())).unique(sorted=True)
            print(f'{corpus} {sequence + 1}/{len(sequences)}: '
                  + ', '.join(f'{k}={len(v)}' for k, v in chosen.items()), flush=True)
            features = collect_readouts(model, layers, ids.unsqueeze(0), positions)
            if args.save_features:
                torch.save({'positions': positions, 'features': features}, out / f'{corpus}_{sequence}_features.pt')
            for name, selected in chosen.items():
                indices = torch.searchsorted(positions, selected)
                for component in COMPONENTS:
                    x = torch.stack([features[layer][component][indices] for layer in layers])
                    measurements = pairwise_similarity(x, args.analysis_device, args.seed + sequence, args.shuffles)
                    for i, layer in enumerate(layers):
                        total_norm = features[layer]['total'][indices].double().norm()
                        magnitudes.append({'corpus': corpus, 'sequence': sequence, 'slice': name,
                            'component': component, 'layer': layer,
                            'frobenius_norm': x[i].double().norm().item(),
                            'relative_to_total_norm': (x[i].double().norm() / total_norm).item() if total_norm else None})
                        for j in range(i, len(layers)):
                            row = {'corpus': corpus, 'sequence': sequence, 'slice': name,
                                   'component': component, 'layer_i': layer, 'layer_j': layers[j],
                                   'n_queries': len(selected)}
                            for metric, values in measurements.items():
                                row[metric] = float(values[i, j]) if torch.isfinite(values[i, j]) else None
                            rows.append(row)
                    del x
            del features
            print(f'{corpus} {sequence + 1}: complete', flush=True)
    if not rows:
        raise RuntimeError('No usable similarity measurements')
    with (out / 'pairs.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    summary = aggregate_pairs(rows)
    for filename, payload in (('summary.json', summary), ('selections.json', selections), ('magnitudes.json', magnitudes)):
        (out / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    if not args.no_plots:
        plot_similarity(summary, out, COMPONENTS)
    if identity != {name: checkpoint_files(path) for name, path in (('ckpt', args.ckpt), ('adapter', args.adapter))}:
        raise RuntimeError('Checkpoint changed during the probe; use fixed checkpoints')
    metadata['complete'] = True
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote similarity results to {out}')


if __name__ == '__main__':
    main()
