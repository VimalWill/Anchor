"""Rank of the learned fast weights, straight from ttt_params.pt.

diag_rank.py answers a different question: it runs a forward pass over a
corpus and measures the rank of states FITTED on real text, chunk by chunk.
That needs a GPU, a model and a dataset. If what you want is whether the
memory a checkpoint carries is low-rank, that is an SVD on tensors already
on disk -- CPU, seconds, no model.

A TTT memory is three matrices per head: w0 and w2 map the input into the
SwiGLU hidden width, w1 maps back out. Each is reported separately, since
they have different shapes and there is no single matrix to call "the state".

Effective rank is the participation ratio (sum s)^2 / sum s^2, which counts
how many directions carry the mass rather than how many are nonzero --
numerical rank would say "full" for a matrix with one dominant direction.
It is reported as a fraction of the maximum, so 1.0 is flat and 0.1 means a
tenth of the directions do the work.

    python rank_state.py --ckpt /path/to/stage2/best_ckpt
"""
import argparse
import json
import re
from pathlib import Path

import torch

PREFIX = re.compile(r'^base_model\.model\.')
KEY = re.compile(r'layers\.(\d+)\..*\.(w0|w1|w2)$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='dir holding ttt_params.pt')
    ap.add_argument('--out', help='write a JSON summary here')
    a = ap.parse_args()

    path = Path(a.ckpt) / 'ttt_params.pt'
    if not path.is_file():
        raise SystemExit(f'no ttt_params.pt in {a.ckpt}')
    state = torch.load(path, map_location='cpu', weights_only=True)

    found = {}
    for key, value in state.items():
        match = KEY.search(PREFIX.sub('', key))
        if match:
            found[(int(match.group(1)), match.group(2))] = value.float()
    if not found:
        raise SystemExit('no w0/w1/w2 tensors; is this a stage-2 checkpoint?')

    layers = sorted({layer for layer, _ in found})
    print(f'{len(layers)} memories in {path}')
    print(f'\n{"layer":>6} {"w":>3} {"shape":>16} {"eff rank":>20} '
          f'{"top-10% mass":>13} {"condition":>10}')
    summary = {}
    for layer in layers:
        for name in ('w0', 'w1', 'w2'):
            weight = found.get((layer, name))
            if weight is None:
                continue
            # [heads, out, in] -> singular values per head
            sv = torch.linalg.svdvals(weight.double())
            full = sv.shape[-1]
            effective = (sv.sum(-1) ** 2 / sv.pow(2).sum(-1)) / full
            ordered = sv.sort(dim=-1, descending=True).values
            top = max(1, full // 10)
            mass = ordered[..., :top].pow(2).sum(-1) / ordered.pow(2).sum(-1)
            kappa = ordered[..., 0] / ordered[..., -1].clamp(min=1e-12)
            print(f'{layer:>6} {name:>3} {str(tuple(weight.shape)):>16} '
                  f'{effective.mean():>8.3f} of full {full:<5} '
                  f'{mass.mean():>12.1%} {kappa.mean():>10.1f}')
            summary.setdefault(str(layer), {})[name] = dict(
                shape=list(weight.shape), full_rank=full,
                effective_rank_fraction=effective.mean().item(),
                top_decile_energy=mass.mean().item(),
                condition=kappa.mean().item())
    print('\neffective rank is the participation ratio over singular values, '
          'as a fraction of full;\n1.0 is a flat spectrum, small means a few '
          'directions carry the memory.')
    if a.out:
        Path(a.out).write_text(json.dumps(dict(ckpt=a.ckpt, memories=summary), indent=2) + '\n')
        print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
