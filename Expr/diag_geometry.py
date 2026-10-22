"""Raw projection geometry and offline crossed projections for TTT or LoLCATs.

No training or memory intervention. Fit on calibration sequences; evaluate on
disjoint sequences. Raw Q/K/V are captured before RoPE and feature maps, with
native KV heads (no GQA repetition). Hidden-state maps use a common train-only
PCA basis; their singular values are not full residual-stream transformations.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diag_rank import checkpoint_files, load_corpora, validate_tokens


def pca(x):
    """Exact centered SVD; covariance eigenvalues equal s**2 / (n-1)."""
    mean = x.mean(0)
    _, s, vh = torch.linalg.svd(x - mean, full_matrices=False)
    tol = max(x.shape) * torch.finfo(x.dtype).eps * s[0]
    return mean, s, vh, int((s > tol).sum())


def relative_error(predicted, target):
    denominator = target.square().sum()
    return float(((predicted - target).square().sum() / denominator).sqrt()) if denominator > 0 else None


def compare(x, y, xt, yt, ridge=1e-4, subspace_rank=16):
    """Row convention y ~= (x - mean_x) @ T + mean_y.

    Orthogonal fits permit reflections. Ridge is scaled by mean eigenvalue
    of X.T X. All centering, maps, and PCA axes use calibration data only.
    """
    if x.shape != y.shape or xt.shape != yt.shape or x.shape[1] != xt.shape[1]:
        raise ValueError('Aligned features must have matching dimensions')
    if not all(torch.isfinite(z).all() for z in (x, y, xt, yt)):
        raise ValueError('Nonfinite features')
    mx, sx, vx, rx = pca(x)
    my, sy, vy, ry = pca(y)
    xc, yc, xv, yv = x - mx, y - my, xt - mx, yt - my
    u, s, vh = torch.linalg.svd(xc.T @ yc, full_matrices=False)
    orthogonal = u @ vh
    power = xc.square().sum()
    scale = float(s.sum() / power) if power > 0 else 0.0
    gram = xc.T @ xc
    penalty = ridge * float(gram.trace() / x.shape[1])
    linear = torch.linalg.solve(gram + max(penalty, torch.finfo(x.dtype).eps)
                                * torch.eye(x.shape[1], device=x.device, dtype=x.dtype), xc.T @ yc)
    lu, ls, lvh = torch.linalg.svd(linear, full_matrices=False)
    rotation = lu @ lvh
    # Row-vector right polar decomposition T = P @ R.
    stretch = (lu * ls) @ lu.T
    r = min(subspace_rank, rx, ry)
    angles = (torch.rad2deg(torch.acos(torch.linalg.svdvals(vx[:r] @ vy[:r].T)
                                    .clamp(0, 1))).tolist() if r else [])
    row = {'n_calibration': len(x), 'n_evaluation': len(xt), 'dimension': x.shape[1],
           'source_rank': rx, 'target_rank': ry, 'subspace_rank': r,
           'principal_angles_degrees': angles,
           'mean_shift_norm': float((my - mx).norm()), 'uniform_scale': scale,
           'ridge_lambda': penalty, 'linear_singular_values': ls.tolist(),
           'linear_orthogonal_determinant': float(torch.linalg.det(rotation)),
           'polar_unique': bool(ls[-1] > ls[0] * x.shape[1] * torch.finfo(x.dtype).eps),
           'source_singular_values': sx.tolist(), 'target_singular_values': sy.tolist(),
           'source_covariance_eigenvalues': (sx.square() / (len(x) - 1)).tolist(),
           'target_covariance_eigenvalues': (sy.square() / (len(y) - 1)).tolist()}
    maps = {'identity': torch.eye(x.shape[1], device=x.device, dtype=x.dtype),
            'orthogonal': orthogonal, 'scaled_orthogonal': scale * orthogonal, 'linear': linear}
    for name, transform in maps.items():
        row[f'{name}_train_error'] = relative_error(xc @ transform, yc)
        row[f'{name}_test_error'] = relative_error(xv @ transform, yv)
    # Predicting just the calibration target mean has normalized error 1.
    row['mean_only_test_error'] = relative_error(torch.zeros_like(yv), yv)
    row['uncentered_identity_test_error'] = relative_error(xt, yt)
    artifacts = {**maps, 'mean_x': mx, 'mean_y': my,
                 'linear_polar_orthogonal': rotation, 'linear_polar_stretch': stretch}
    return row, {k: v.cpu() for k, v in artifacts.items()}


@torch.no_grad()
def capture(backbone, layers, ids, positions, pairs):
    """Capture true linear inputs/outputs, then cross only sampled inputs."""
    result, hooks = {}, []

    def hook(layer, kind):
        def save(module, inputs, output):
            h = inputs[0][0].index_select(0, positions.to(inputs[0].device)).detach().float().cpu()
            values = result.setdefault(layer, {})
            if 'hidden' in values and not torch.equal(values['hidden'], h):
                raise ValueError('Q/K/V do not share the same input; unsupported attention implementation')
            values['hidden'] = h
            values[kind] = output[0].index_select(0, positions.to(output.device)).detach().float().cpu()
        return save

    try:
        for layer, attn in layers.items():
            for kind in ('q', 'k', 'v'):
                projection = getattr(attn, kind + '_proj')
                if not isinstance(projection, torch.nn.Linear):
                    raise ValueError('Expected merged nn.Linear Q/K/V projections')
                hooks.append(projection.register_forward_hook(hook(layer, kind)))
        backbone(input_ids=ids.to(next(backbone.parameters()).device), use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    if set(result) != set(layers) or any(set(v) != {'hidden', 'q', 'k', 'v'} for v in result.values()):
        raise RuntimeError('Not all Q/K/V hooks ran')
    crossed = {}
    for i, j in pairs:
        for source, destination in ((i, j), (j, i)):
            for kind in ('q', 'k', 'v'):
                projection = getattr(layers[destination], kind + '_proj')
                h = result[source]['hidden'].to(projection.weight)
                crossed[(source, destination, kind)] = F.linear(
                    h, projection.weight, projection.bias).float().cpu()
    return result, crossed


def load_checkpoint(args):
    from transformers import AutoModelForCausalLM
    if args.model_kind == 'lolcats':
        sys.path.append(str(ROOT / 'third_party' / 'Liger'))
        import fla  # noqa: F401
        import lolcats  # noqa: F401
        model = AutoModelForCausalLM.from_pretrained(
            args.ckpt, torch_dtype=torch.bfloat16, device_map='cuda')
        if args.adapter:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    else:
        from omegaconf import OmegaConf
        from Training.train import build_model_config
        from eval import load_model
        from LinearTTT.model.LinearizeLlama.LinearizeLlama import use_sdpa_sliding_window
        from LinearTTT.model.LinearizeMistral.LinearizeMistral import use_sdpa_sliding_window as mistral_sdpa
        config = OmegaConf.load(args.cfg)  # Do not resolve unused training settings.
        config.model.pretrained_model_name_or_path = args.ckpt
        config.model.max_length = args.seq_len
        model_config = build_model_config(config)
        model_config.validate_ttt()
        model = load_model(args.ckpt, model_config, args.adapter)
        use_sdpa_sliding_window(True)
        mistral_sdpa(True)
    return model.eval().requires_grad_(False)


def parse_pair(value):
    try:
        i, j = map(int, value.split(':'))
        if i < 0 or j < 0 or i == j:
            raise ValueError
        return i, j
    except ValueError:
        raise argparse.ArgumentTypeError('Use two distinct nonnegative layers, e.g. 0:1')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model-kind', choices=['ttt', 'lolcats'], default='ttt')
    ap.add_argument('--cfg', help='Required for TTT; use the checkpoint evaluation config')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--adapter')
    ap.add_argument('--base', required=True, help='Tokenizer identifier, same for paired model comparisons')
    ap.add_argument('--pairs', nargs='+', type=parse_pair, required=True)
    ap.add_argument('--heads', nargs='+', type=int, default=[0], help='Native Q/K/V head indices; default head 0')
    ap.add_argument('--corpora', nargs='+', choices=['pg19', 'wikitext'], default=['pg19', 'wikitext'])
    ap.add_argument('--pg19-path', default='emozilla/pg19')
    ap.add_argument('--seq-len', type=int, default=8192)
    ap.add_argument('--seqs', type=int, default=8)
    ap.add_argument('--fit-seqs', type=int, default=6)
    ap.add_argument('--samples', type=int, default=256)
    ap.add_argument('--start-position', type=int, default=512)
    ap.add_argument('--hidden-rank', type=int, default=64)
    ap.add_argument('--subspace-rank', type=int, default=16)
    ap.add_argument('--ridge', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--analysis-device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--tokens-file')
    ap.add_argument('--save-features', action='store_true')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    if (not 1 <= args.fit_seqs < args.seqs or args.samples < 2 or
            not 0 <= args.start_position < args.seq_len or args.ridge <= 0 or
            args.hidden_rank < 1 or args.subspace_rank < 1 or min(args.heads) < 0):
        ap.error('Invalid split, sample count, position, rank, ridge, or head index')
    if args.model_kind == 'ttt' and not args.cfg:
        ap.error('--cfg is required for TTT')
    if not torch.cuda.is_available():
        ap.error('Checkpoint evaluation requires CUDA; mathematical analysis also supports CPU')
    args.pairs = list(dict.fromkeys(args.pairs))
    args.heads = list(dict.fromkeys(args.heads))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    if args.tokens_file:
        saved = torch.load(args.tokens_file, map_location='cpu', weights_only=True)
        if saved['tokenizer'] != args.base:
            raise ValueError('Token file tokenizer must match --base exactly')
        corpora = {name: saved['corpora'][name] for name in dict.fromkeys(args.corpora)}
        sources = {name: saved['sources'][name] for name in corpora}
    else:
        corpora, sources = load_corpora(args, AutoTokenizer.from_pretrained(args.base))
    identity = {name: checkpoint_files(path) for name, path in
                (('ckpt', args.ckpt), ('adapter', args.adapter))}
    model = load_checkpoint(args)
    validate_tokens(corpora, args.seq_len, args.seqs, model.config.vocab_size)
    wanted = sorted({i for pair in args.pairs for i in pair})
    if wanted[-1] >= len(model.model.layers):
        ap.error('Requested layer outside decoder')
    layers = {i: model.model.layers[i].self_attn for i in wanted}
    dimensions = {i: layers[i].head_dim for i in wanted}
    for i, attn in layers.items():
        for kind in ('q', 'k', 'v'):
            if (max(args.heads) + 1) * dimensions[i] > getattr(attn, kind + '_proj').weight.shape[0]:
                ap.error(f'Head unavailable in native {kind.upper()} projection at layer {i}')
    metadata = {'complete': False, 'arguments': vars(args), 'sources': sources,
                'model_config': model.config.to_dict(), 'checkpoint_files': identity,
                'code_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'torch_version': str(torch.__version__),
                'token_hashes': {n: hashlib.sha256(t.numpy().tobytes()).hexdigest() for n, t in corpora.items()},
                'limitations': ['Offline crossed projections are not whole-model interventions.',
                    'No Gaussian assumption; no recurrent-state capture.',
                    'Raw projections precede RoPE, GQA repetition, and feature maps.',
                    'Hidden maps use a common calibration PCA subspace, without whitening.',
                    'Orthogonal maps can include reflections; rank-deficient polar factors are not unique.',
                    'Linear singular values depend on ridge regularization and feature coordinates.',
                    'Sequence split: PG19 book prefixes; WikiText contiguous token blocks.']}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    torch.save({'tokenizer': args.base, 'corpora': corpora, 'sources': sources}, out / 'tokens.pt')
    generator = torch.Generator().manual_seed(args.seed)
    positions = (torch.randperm(args.seq_len - args.start_position, generator=generator)
                 [:args.samples] + args.start_position).sort().values
    metadata['positions'] = positions.tolist()
    all_features = {}
    for corpus, sequences in corpora.items():
        collected = []
        for n, ids in enumerate(sequences):
            print(f'{corpus}: capture {n + 1}/{len(sequences)}', flush=True)
            collected.append(capture(model.model, layers, ids.unsqueeze(0), positions, args.pairs))
        all_features[corpus] = collected
    # Release model memory before factorization.
    del layers, attn, model
    torch.cuda.empty_cache()
    if args.save_features:
        torch.save(all_features, out / 'features.pt')
    rows = []
    with (out / 'spectra.jsonl').open('w') as stream:
        for corpus, collected in all_features.items():
            def get(source, weight=None, kind='hidden'):
                values = [a[source][kind] if weight is None or source == weight
                          else b[(source, weight, kind)] for a, b in collected]
                return (torch.cat(values[:args.fit_seqs]).to(args.analysis_device, torch.float64),
                        torch.cat(values[args.fit_seqs:]).to(args.analysis_device, torch.float64))

            def record(label, kind, head, x, y, xt, yt, extra=None):
                result, artifacts = compare(x, y, xt, yt, args.ridge, args.subspace_rank)
                result.update(corpus=corpus, layer_i=i, layer_j=j, comparison=label, feature=kind, head=head)
                if extra:
                    result.update(extra)
                stream.write(json.dumps(result, allow_nan=False) + '\n')
                stream.flush()
                rows.append({k: v for k, v in result.items() if not isinstance(v, list)})
                torch.save(artifacts, out / f'{corpus}_L{i}_L{j}_{kind}_h{head}_{label}.pt')

            for i, j in args.pairs:
                print(f'{corpus}: analyze L{i} -> L{j}', flush=True)
                hi, hit = get(i)
                hj, hjt = get(j)
                pooled = torch.cat((hi, hj))
                center, s, axes, rank = pca(pooled)
                r = min(args.hidden_rank, rank)
                if r < 1:
                    raise ValueError('Hidden-state calibration data has zero variance')
                basis = axes[:r].T
                torch.save({'center': center.cpu(), 'basis': basis.cpu(),
                            'singular_values': s.cpu(),
                            'covariance_eigenvalues': (s.square() / (len(pooled) - 1)).cpu()},
                           out / f'{corpus}_L{i}_L{j}_hidden_basis.pt')
                record('observed', 'hidden_pca', -1, (hi-center) @ basis, (hj-center) @ basis,
                       (hit-center) @ basis, (hjt-center) @ basis,
                       {'pooled_pca_energy_retained': float(s[:r].square().sum() / s.square().sum())})
                del hi, hj, hit, hjt, pooled, axes, basis
                for kind in ('q', 'k', 'v'):
                    cells = {(h, w): get(h, w, kind) for h, w in ((i, i), (i, j), (j, i), (j, j))}
                    contrasts = [('observed', (i, i), (j, j)),
                                 ('weights_at_hi', (i, i), (i, j)),
                                 ('weights_at_hj', (j, i), (j, j)),
                                 ('hidden_at_wi', (i, i), (j, i)),
                                 ('hidden_at_wj', (i, j), (j, j))]
                    if dimensions[i] != dimensions[j]:
                        raise ValueError('Cross-layer head dimensions must match')
                    d = dimensions[i]
                    for head in args.heads:
                        for label, a, b in contrasts:
                            x, xt = (z[:, head*d:(head+1)*d] for z in cells[a])
                            y, yt = (z[:, head*d:(head+1)*d] for z in cells[b])
                            record(label, kind, head, x, y, xt, yt)
                    del cells
    with (out / 'summary.csv').open('w', newline='') as f:
        fields = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    if identity != {name: checkpoint_files(path) for name, path in
                    (('ckpt', args.ckpt), ('adapter', args.adapter))}:
        raise RuntimeError('Checkpoint files changed during analysis')
    metadata['complete'] = True
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Wrote {len(rows)} comparisons to {out}', flush=True)


if __name__ == '__main__':
    main()
