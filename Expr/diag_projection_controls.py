"""Every-layer projection controls: actual, ZCA-whitened, shared N(0,I).

Calibrate whitening on the first fit-seqs sequences only. Native Q/K/V heads,
before RoPE and feature maps. Compare adjacent selected layers on held-out
paired samples. Exact WW.T is the identity-covariance baseline. Full whitening
is required unless --allow-partial-whitening is explicitly selected.
"""
import argparse
import csv
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from diag_geometry import compare, load_checkpoint
from diag_rank import checkpoint_files, load_corpora, validate_tokens


def covariance(x):
    centered = x - x.mean(0)
    return centered.T @ centered / (len(x) - 1)


def whiten_fit(x, rtol=1e-8, allow_partial=False):
    """Symmetric ZCA, preserving ambient coordinates; never ridge-claim identity.

    In partial mode Cov(Z) is the supported-subspace projector, not I.
    Returned basis spans that supported subspace for exact W P W.T controls.
    """
    if len(x) < 2 or not torch.isfinite(x).all():
        raise ValueError('Whitening needs at least two finite calibration samples')
    mean = x.mean(0)
    c = covariance(x)
    eigenvalues, axes = torch.linalg.eigh(c)
    threshold = max(float(eigenvalues[-1]) * rtol, torch.finfo(x.dtype).tiny)
    keep = eigenvalues > threshold
    rank = int(keep.sum())
    if rank == 0:
        raise ValueError('Hidden inputs have zero supported variance')
    if rank != x.shape[1] and not allow_partial:
        raise ValueError(f'Whitening rank {rank}/{x.shape[1]} from {len(x)} samples. '
                         'Increase --samples/--fit-seqs or explicitly use --allow-partial-whitening. '
                         'A truncated whitening is not an identity-covariance control.')
    basis = axes[:, keep]
    transform = (basis * eigenvalues[keep].rsqrt()) @ basis.T
    z = (x - mean) @ transform
    projector = basis @ basis.T
    residual = float((covariance(z) - projector).norm() / projector.norm())
    info = {'input_dimension': x.shape[1], 'calibration_samples': len(x), 'whitening_rank': rank,
            'full_identity': rank == x.shape[1], 'relative_threshold': rtol,
            'retained_condition_number': float(eigenvalues[keep][-1] / eigenvalues[keep][0]),
            'calibration_covariance_relative_error': residual,
            'hidden_covariance_eigenvalues': eigenvalues.flip(0).clamp_min(0).tolist()}
    return mean, transform, basis, info


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
            torch.save(sampled, directory / f'L{index}_s{sequence}.pt')
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


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k, value in row.items() if not isinstance(value, list)))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: v for k, v in row.items() if k in fields} for row in rows)


def analyze(args, specs, weights, corpora, cache, out):
    device = args.analysis_device
    layers = sorted(specs)
    fit_count = args.fit_seqs * args.samples
    count = args.seqs * args.samples
    width = specs[layers[0]]['width']
    # CPU RNG makes the exact inputs independent of the analysis device.
    rng = torch.Generator().manual_seed(args.seed + 1000)
    gaussian = torch.randn(count, width, generator=rng, dtype=torch.float64).to(device)
    layer_rows, pair_rows, whitening_rows = [], [], []
    with (out / 'layer_spectra.jsonl').open('w') as spectra, (out / 'pair_spectra.jsonl').open('w') as pairs:
        for corpus in corpora:
            previous, previous_covs, previous_layer = None, None, None
            for layer in layers:
                started = time.monotonic()
                print(f'{corpus} L{layer}: whitening and projections', flush=True)
                hidden = torch.cat([torch.load(cache / corpus / f'L{layer}_s{s}.pt',
                                               weights_only=True, map_location='cpu') for s in range(args.seqs)])
                hidden = hidden.to(device, torch.float64)
                mean, zca, basis, info = whiten_fit(hidden[:fit_count], args.whitening_rtol,
                                                   args.allow_partial_whitening)
                info.update(corpus=corpus, layer=layer)
                whitening_rows.append(info)
                with (out / f'{corpus}_L{layer}_whitening.json').open('w') as f:
                    json.dump(info, f, allow_nan=False)
                transformed = (hidden - mean) @ zca
                inputs = {'actual': hidden, 'whitened': transformed, 'gaussian': gaussian}
                current, covs, artifacts = {}, {}, {}
                for kind, head_specs in weights[layer].items():
                    for head, (weight, bias) in head_specs.items():
                        w = weight.to(device, torch.float64)
                        b = bias.to(device, torch.float64) if bias is not None else None
                        exact = w @ w.T
                        wp = w @ basis
                        supported = wp @ wp.T
                        artifacts[f'{kind}_h{head}_exact_identity'] = exact.cpu()
                        artifacts[f'{kind}_h{head}_exact_supported'] = supported.cpu()
                        for control, c in [('identity_exact', exact), ('supported_exact', supported)]:
                            row = dict(corpus=corpus, layer=layer, feature=kind, head=head,
                                       control=control, split='population', **covariance_metrics(c))
                            spectra.write(json.dumps(row, allow_nan=False) + '\n')
                            layer_rows.append(row)
                            covs[(kind, head, control, 'population')] = c.cpu()
                        for control, h in inputs.items():
                            y = F.linear(h, w, b)
                            current[(kind, head, control)] = y.cpu()
                            for split, v in [('calibration', y[:fit_count]), ('evaluation', y[fit_count:])]:
                                c = covariance(v)
                                baseline = supported if control == 'whitened' else exact
                                row = dict(corpus=corpus, layer=layer, feature=kind, head=head,
                                           control=control, split=split,
                                           reference='supported_exact' if control == 'whitened' else 'identity_exact',
                                           mean_norm=float(v.mean(0).norm()),
                                           **covariance_metrics(c), **covariance_distance(baseline, c))
                                spectra.write(json.dumps(row, allow_nan=False) + '\n')
                                layer_rows.append(row)
                                covs[(kind, head, control, split)] = c.cpu()
                                artifacts[f'{kind}_h{head}_{control}_{split}'] = c.cpu()
                        if previous is not None:
                            for control in inputs:
                                key = (kind, head, control)
                                x, y = previous[key].to(device), current[key].to(device)
                                row, maps = compare(x[:fit_count], y[:fit_count], x[fit_count:], y[fit_count:],
                                                    args.ridge, args.subspace_rank)
                                row.update(corpus=corpus, layer_i=previous_layer, layer_j=layer,
                                           feature=kind, head=head, control=control)
                                for split in ('calibration', 'evaluation'):
                                    key_c = (kind, head, control, split)
                                    distances = covariance_distance(previous_covs[key_c], covs[key_c])
                                    row.update({f'{split}_{k}': value for k, value in distances.items()})
                                pairs.write(json.dumps(row, allow_nan=False) + '\n')
                                pair_rows.append(row)
                                if args.save_maps:
                                    torch.save(maps, out / f'{corpus}_L{previous_layer}_L{layer}_{kind}_h{head}_{control}_maps.pt')
                            for control in ('identity_exact', 'supported_exact'):
                                key_c = (kind, head, control, 'population')
                                row = dict(corpus=corpus, layer_i=previous_layer, layer_j=layer,
                                           feature=kind, head=head, control=control,
                                           **covariance_distance(previous_covs[key_c], covs[key_c]))
                                pairs.write(json.dumps(row, allow_nan=False) + '\n')
                                pair_rows.append(row)
                torch.save(artifacts, out / f'{corpus}_L{layer}_covariances.pt')
                previous, previous_covs, previous_layer = current, covs, layer
                spectra.flush()
                pairs.flush()
                del inputs, hidden, transformed, zca, basis
                print(f'{corpus} L{layer}: complete in {time.monotonic()-started:.1f}s', flush=True)
    write_csv(out / 'layer_summary.csv', layer_rows)
    write_csv(out / 'pair_summary.csv', pair_rows)
    write_csv(out / 'whitening_summary.csv', whitening_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-kind', choices=['ttt', 'lolcats'], default='ttt')
    ap.add_argument('--cfg')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--base', required=True)
    ap.add_argument('--layers', nargs='+', type=int, help='Default: EVERY decoder layer; pairs are adjacent selected layers')
    group = ap.add_mutually_exclusive_group()
    group.add_argument('--heads', nargs='+', type=int, default=[0], help='Native heads; default head 0 at every layer')
    group.add_argument('--all-heads', action='store_true', help='All native heads, independently for Q and KV')
    ap.add_argument('--corpora', nargs='+', choices=['pg19', 'wikitext'], default=['pg19', 'wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--fit-seqs', type=int, default=6)
    ap.add_argument('--samples', type=int, default=1024)
    ap.add_argument('--start-position', type=int, default=512)
    ap.add_argument('--whitening-rtol', type=float, default=1e-8)
    ap.add_argument('--allow-partial-whitening', action='store_true')
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--subspace-rank', type=int, default=16)
    ap.add_argument('--analysis-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tokens-file')
    ap.add_argument('--save-maps', action='store_true')
    ap.add_argument('--out', required=True, help='New directory; temporary hidden inputs are stored here, then removed')
    args = ap.parse_args()
    if (not 1 <= args.fit_seqs < args.seqs or args.samples < 2 or args.ridge <= 0 or
            not 0 < args.whitening_rtol < 1 or args.subspace_rank < 1 or
            not 0 <= args.start_position < args.seq_len or
            args.samples > args.seq_len - args.start_position or min(args.heads) < 0):
        ap.error('Invalid split, sample count, ridge, rank, threshold, heads, or positions')
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
            weights[i][kind] = {h: (proj.weight[h*d:(h+1)*d].detach().float().cpu().clone(),
                                     proj.bias[h*d:(h+1)*d].detach().float().cpu().clone()
                                     if proj.bias is not None else None) for h in heads}
    if len({(s['width'], s['head_dim']) for s in specs.values()}) != 1:
        raise ValueError('Selected layers must share hidden/head dimensions')
    if any({k: set(v) for k, v in weights[i].items()} !=
           {k: set(v) for k, v in weights[chosen[0]].items()} for i in chosen):
        raise ValueError('Selected layers must have matching native head counts')
    if not args.allow_partial_whitening and args.fit_seqs * args.samples <= specs[chosen[0]]['width']:
        ap.error('Full whitening needs more calibration samples than hidden dimensions')
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
                'definitions': {'identity_exact': 'WW.T for population input covariance I',
                    'supported_exact': 'W P W.T; P is the retained calibration hidden subspace projector',
                    'whitened': 'Real hidden inputs centered/ZCA-transformed using calibration statistics only',
                    'gaussian': 'Same standard Gaussian input samples at every layer; independent fit/test rows',
                    'actual': 'Captured projection inputs multiplied by merged weights in float64'},
                'limitations': ['No recurrent-state capture or causal attribution percentages.',
                    'Whitening does not Gaussianize real inputs; held-out covariance need not be identity.',
                    'Population identity covariance cannot reveal orthogonal rotations of isotropic inputs.',
                    'Paired-output fits quantify predictability, not size of a causal effect.',
                    'Raw projection geometry precedes RoPE and nonlinear feature processing.',
                    'Sequence split: PG19 books, WikiText token blocks; no uncertainty intervals.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch.save({'tokenizer': args.base, 'corpora': corpora, 'sources': sources}, out / 'tokens.pt')
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
