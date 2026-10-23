"""Per-layer Q/K/V weight, hidden-state, and projected-covariance analysis.

For each layer/head, compare W Sigma_x W.T (actual hidden inputs) with W W.T
(isotropic Gaussian inputs), then test whether the predicted covariance
reproduces the empirical projected covariance and its principal directions.
Arrays are saved only as .npy; summaries and metadata are JSON/JSONL.
"""
import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from diag_geometry import load_checkpoint
from diag_rank import checkpoint_files, load_corpora, validate_tokens


def covariance(x):
    centered = x - x.mean(0)
    return centered.T @ centered / (len(x) - 1)


def covariance_metrics(c):
    values = torch.linalg.eigvalsh(c).flip(0).clamp_min(0)
    total = float(values.sum())
    out = {'covariance_eigenvalues': values.tolist(), 'trace': total,
           'stable_rank': float(total / values[0]) if total else None}
    for threshold in (90, 95, 99):
        out[f'r{threshold}'] = (min(int(torch.searchsorted(values.cumsum(0), total * threshold / 100)) + 1,
                                  len(values)) if total else 0)
    return out


def covariance_distance(a, b):
    """Scale-sensitive and trace-normalized differences, in native coordinates."""
    na, ta, tb = float(a.norm()), float(a.trace()), float(b.trace())
    return {'relative_covariance_difference': float((b-a).norm()) / na if na else None,
            'trace_ratio': tb / ta if ta else None,
            'trace_normalized_difference': float((a/ta - b/tb).norm()) if ta and tb else None}


def subspace_overlap(a, b, rank):
    """Mean squared cosine overlap between leading covariance eigenspaces."""
    _, ua = torch.linalg.eigh(a)
    _, ub = torch.linalg.eigh(b)
    ua, ub = ua[:, -rank:], ub[:, -rank:]
    return float((ua.T @ ub).square().sum() / rank)


def distribution_summary(x):
    x = x.reshape(-1).double()
    mean = x.mean()
    centered = x - mean
    variance = centered.square().mean()
    std = variance.sqrt()
    z = centered / std.clamp_min(torch.finfo(x.dtype).tiny)
    return {'count': int(x.numel()), 'mean': float(mean), 'std': float(std),
            'min': float(x.min()), 'q01': float(torch.quantile(x, .01)),
            'q25': float(torch.quantile(x, .25)), 'median': float(torch.quantile(x, .5)),
            'q75': float(torch.quantile(x, .75)), 'q99': float(torch.quantile(x, .99)),
            'max': float(x.max()), 'skewness': float(z.pow(3).mean()),
            'excess_kurtosis': float(z.pow(4).mean() - 3)}


@torch.no_grad()
def capture_inputs(backbone, layers, ids, positions, directory, sequence):
    """Stream one sampled hidden tensor per layer to temporary disk storage."""
    hooks, seen = [], set()

    def hook(index):
        def save(module, inputs):
            if index in seen:
                raise RuntimeError(f'Projection invoked more than once at layer {index}')
            seen.add(index)
            h = inputs[0]
            if h.ndim != 3 or h.shape[0] != 1:
                raise ValueError('Expected projection input [1, tokens, hidden]')
            sampled = h[0].index_select(0, positions.to(h.device)).detach().float().cpu()
            np.save(directory / f'L{index}_s{sequence}.npy', sampled.numpy())
        return save

    try:
        for i, attn in layers.items():
            hooks.append(attn.q_proj.register_forward_pre_hook(hook(i)))
        backbone(input_ids=ids.to(next(backbone.parameters()).device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    if seen != set(layers):
        raise RuntimeError('Some layer input hooks did not run')


def analyze(args, specs, weights, corpora, cache, out):
    device = args.analysis_device
    layers = sorted(specs)
    count = args.seqs * args.samples
    width = specs[layers[0]]['width']
    # CPU RNG makes the exact inputs independent of the analysis device.
    rng = torch.Generator().manual_seed(args.seed + 1000)
    gaussian = torch.randn(count, width, generator=rng, dtype=torch.float64).to(device)
    with (out / 'layer_spectra.jsonl').open('w') as spectra:
        for corpus in corpora:
            for layer in layers:
                started = time.monotonic()
                print(f'{corpus} L{layer}: hidden covariance and projections', flush=True)
                hidden_np = np.concatenate([np.load(cache / corpus / f'L{layer}_s{s}.npy', mmap_mode='r')
                                            for s in range(args.seqs)], axis=0)
                np.save(out / f'{corpus}_L{layer}_hidden.npy', hidden_np)
                hidden = torch.from_numpy(np.asarray(hidden_np)).to(device, torch.float64)
                hidden_cov = covariance(hidden)
                hidden_mean = hidden.mean(0)
                hidden_summary = {'corpus': corpus, 'layer': layer,
                    'mean_norm': float(hidden_mean.norm()),
                    'coordinate_distribution': distribution_summary(hidden),
                    'covariance': covariance_metrics(hidden_cov)}
                with (out / f'{corpus}_L{layer}_hidden.json').open('w') as f:
                    json.dump(hidden_summary, f, allow_nan=False)
                for kind, head_specs in weights[layer].items():
                    for head, (weight, bias) in head_specs.items():
                        w = weight.to(device, torch.float64)
                        exact = w @ w.T
                        predicted = w @ hidden_cov @ w.T
                        y = hidden @ w.T
                        empirical = covariance(y)
                        g_out = gaussian @ w.T
                        gaussian_cov = covariance(g_out)
                        eval_weight = torch.linalg.eigvalsh(exact).flip(0).clamp_min(0)
                        eval_actual = torch.linalg.eigvalsh(empirical).flip(0).clamp_min(0)
                        row = {'corpus': corpus, 'layer': layer, 'projection': kind,
                            'head': head, 'input_samples': len(hidden),
                            'weight_elements': distribution_summary(weight),
                            'weight_only_output_covariance': covariance_metrics(exact),
                            'gaussian_output_covariance': covariance_metrics(gaussian_cov),
                            'actual_output_covariance': covariance_metrics(empirical),
                            'predicted_W_Sigma_WT': covariance_metrics(predicted),
                            'weight_only_singular_values': eval_weight.sqrt().tolist(),
                            'actual_output_singular_values': eval_actual.sqrt().tolist(),
                            'actual_vs_weight_spectral_stretch_ratio':
                                (eval_actual.sqrt() / eval_weight.sqrt().clamp_min(1e-30)).tolist(),
                            'prediction_vs_empirical': covariance_distance(predicted, empirical),
                            'weight_vs_actual_top_subspace_overlap': {
                                str(k): subspace_overlap(exact, empirical, min(k, len(exact)))
                                for k in (8, 16, 32, 64)},
                            'weight_vs_gaussian_top_subspace_overlap': {
                                str(k): subspace_overlap(exact, gaussian_cov, min(k, len(exact)))
                                for k in (8, 16, 32, 64)},
                            'weight_vs_actual_top_eigenvalue_ratios': {
                                str(k): float((torch.linalg.eigvalsh(empirical).flip(0)[:k].sum() /
                                               torch.linalg.eigvalsh(exact).flip(0)[:k].sum()).item())
                                for k in (8, 16, 32, 64)},
                            'interpretation_note': 'Covariance eigenspace changes are predicted by W Sigma_hidden W.T; this is a second-moment explanation, not proof of a causal mechanism.'}
                        spectra.write(json.dumps(row, allow_nan=False) + '\n')
                spectra.flush()
                del hidden, hidden_np, hidden_cov
                print(f'{corpus} L{layer}: complete in {time.monotonic()-started:.1f}s', flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-kind', choices=['ttt', 'lolcats'], default='ttt')
    ap.add_argument('--cfg')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--base', required=True)
    ap.add_argument('--layers', nargs='+', type=int, help='Default: every decoder layer')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--heads', nargs='+', type=int, default=[0], help='Native heads; default head 0 at every layer')
    group.add_argument('--all-heads', action='store_true', help='All native heads, independently for Q and KV')
    ap.add_argument('--corpora', nargs='+', choices=['pg19', 'wikitext'], default=['pg19', 'wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--samples', type=int, default=1024)
    ap.add_argument('--start-position', type=int, default=512)
    ap.add_argument('--analysis-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tokens-file')
    ap.add_argument('--out', required=True, help='New directory for JSON summaries and NPY arrays')
    args = ap.parse_args()
    if (args.seqs < 1 or args.samples < 2 or
            not 0 <= args.start_position < args.seq_len or
            args.samples > args.seq_len - args.start_position or
            (not args.all_heads and (not args.heads or min(args.heads) < 0))):
        ap.error('Invalid split, sample count, heads, or positions')
    if args.model_kind == 'ttt' and not args.cfg:
        ap.error('--cfg is required for TTT')
    if not torch.cuda.is_available():
        ap.error('Model evaluation requires CUDA')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    if args.tokens_file:
        saved = torch.load(args.tokens_file, map_location='cpu', weights_only=True)
        if saved['tokenizer'] != args.base:
            raise ValueError('Token-file tokenizer must match --base')
        corpora = {n: saved['corpora'][n] for n in dict.fromkeys(args.corpora)}
        sources = {n: saved['sources'][n] for n in corpora}
    else:
        corpora, sources = load_corpora(args, AutoTokenizer.from_pretrained(args.base))
    identity = {n: checkpoint_files(p) for n, p in (('ckpt', args.ckpt), ('adapter', args.adapter))}
    model = load_checkpoint(args)
    validate_tokens(corpora, args.seq_len, args.seqs, model.config.vocab_size)
    chosen = sorted(set(args.layers)) if args.layers is not None else list(range(len(model.model.layers)))
    if not chosen or chosen[0] < 0 or chosen[-1] >= len(model.model.layers):
        ap.error('Invalid layer selection')
    layers = {i: model.model.layers[i].self_attn for i in chosen}
    weights, specs = {}, {}
    for i, attn in layers.items():
        specs[i] = {'width': attn.q_proj.in_features, 'head_dim': attn.head_dim}
        weights[i] = {}
        for kind in ('q', 'k', 'v'):
            proj = getattr(attn, kind + '_proj')
            if not isinstance(proj, torch.nn.Linear) or proj.in_features != specs[i]['width']:
                raise ValueError('Expected merged Q/K/V linear projections sharing input dimensions')
            d = attn.head_dim
            if proj.out_features % d:
                raise ValueError('Projection output is not a whole number of heads')
            heads = list(range(proj.out_features // d)) if args.all_heads else sorted(set(args.heads))
            if max(heads) >= proj.out_features // d:
                ap.error(f'Invalid {kind} head at layer {i}')
            np.save(out / f'L{i}_{kind}_weight.npy', proj.weight.detach().float().cpu().numpy())
            weights[i][kind] = {h: (proj.weight[h*d:(h+1)*d].detach().float().cpu().clone(),
                                     proj.bias[h*d:(h+1)*d].detach().float().cpu().clone()
                                     if proj.bias is not None else None) for h in heads}
    if len({(s['width'], s['head_dim']) for s in specs.values()}) != 1:
        raise ValueError('Selected layers must share hidden/head dimensions')
    if any({k: set(v) for k, v in weights[i].items()} !=
           {k: set(v) for k, v in weights[chosen[0]].items()} for i in chosen):
        raise ValueError('Selected layers must have matching native head counts')
    rng = torch.Generator().manual_seed(args.seed)
    positions = (torch.randperm(args.seq_len - args.start_position, generator=rng)[:args.samples]
                 + args.start_position).sort().values
    metadata = {'complete': False, 'arguments': vars(args), 'layers': chosen,
                'heads': {k: list(v) for k, v in weights[chosen[0]].items()},
                'model_config': model.config.to_dict(), 'sources': sources,
                'checkpoint_files': identity, 'torch_version': str(torch.__version__),
                'positions': positions.tolist(),
                'token_hashes': {n: hashlib.sha256(t.numpy().tobytes()).hexdigest() for n, t in corpora.items()},
                'code_hashes': {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                for n in ('diag_projection_controls.py', 'diag_geometry.py', 'diag_rank.py')},
                'definitions': {'actual_covariance': 'Empirical covariance of x @ W.T',
                    'predicted_covariance': 'W @ Cov(x) @ W.T',
                    'weight_only_covariance': 'W @ W.T, population covariance for standard Gaussian x',
                    'gaussian_control': 'Shared standard Gaussian input multiplied as x_gass @ W.T'},
                'limitations': ['No recurrent-state capture or causal attribution percentages.',
                    'Covariance decomposition verifies a second-moment prediction, not a causal mechanism.',
                    'Raw projection geometry precedes RoPE and nonlinear feature processing.',
                    'Sequence split: PG19 books, WikiText token blocks; no uncertainty intervals.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='.hidden_inputs_', dir=out) as temp:
        cache = Path(temp)
        for corpus, sequences in corpora.items():
            (cache / corpus).mkdir()
            for s, ids in enumerate(sequences):
                print(f'{corpus}: capture every selected layer, sequence {s+1}/{len(sequences)}', flush=True)
                capture_inputs(model.model, layers, ids.unsqueeze(0), positions, cache / corpus, s)
        del model, layers, attn, proj
        torch.cuda.empty_cache()
        analyze(args, specs, weights, corpora, cache, out)
    if identity != {n: checkpoint_files(p) for n, p in (('ckpt', args.ckpt), ('adapter', args.adapter))}:
        raise RuntimeError('Checkpoint files changed during analysis')
    metadata['complete'] = True
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Completed every selected layer; results in {out}', flush=True)


if __name__ == '__main__':
    main()
