"""CPU mathematical and end-to-end checks; no checkpoint downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diag_projection_controls import analyze, capture_inputs, covariance, whiten_fit
from diag_geometry import compare


class ControlsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_full_whitening_and_projected_baseline(self):
        x = torch.randn(300, 8, dtype=torch.float64) * torch.arange(1, 9) + 5
        mean, zca, basis, info = whiten_fit(x)
        z = (x-mean) @ zca
        eye = torch.eye(8, dtype=x.dtype)
        self.assertTrue(info['full_identity'])
        self.assertTrue(torch.allclose(covariance(z), eye, atol=1e-10))
        w = torch.randn(3, 8, dtype=x.dtype)
        self.assertTrue(torch.allclose(covariance(z @ w.T), w @ w.T, atol=1e-10))
        self.assertTrue(torch.allclose(basis @ basis.T, eye, atol=1e-10))

    def test_rank_deficiency_requires_explicit_opt_in(self):
        x = torch.randn(5, 8, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, 'Whitening rank'):
            whiten_fit(x)
        mean, zca, basis, info = whiten_fit(x, allow_partial=True)
        self.assertFalse(info['full_identity'])
        self.assertEqual(info['whitening_rank'], 4)
        self.assertTrue(torch.allclose(covariance((x-mean) @ zca), basis @ basis.T, atol=1e-10))
        w = torch.randn(3, 8, dtype=x.dtype)
        self.assertTrue(torch.allclose(covariance((x-mean) @ zca @ w.T),
                                       (w @ basis) @ (w @ basis).T, atol=1e-10))

    def test_heldout_is_not_rewhitened(self):
        x = torch.randn(300, 4, dtype=torch.float64)
        mean, zca, _, _ = whiten_fit(x)
        test = 4 * x + 20
        self.assertTrue(torch.allclose(covariance((test-mean) @ zca),
                                       16 * torch.eye(4, dtype=x.dtype), atol=1e-10))

    def test_shared_gaussian_detects_rotation_invisible_in_population_covariance(self):
        x = torch.randn(1000, 4, dtype=torch.float64)
        r = torch.linalg.qr(torch.randn(4, 4, dtype=x.dtype))[0]
        self.assertTrue(torch.allclose(r @ r.T, torch.eye(4, dtype=x.dtype), atol=1e-10))
        row, _ = compare(x[:700], x[:700] @ r, x[700:], x[700:] @ r)
        self.assertGreater(row['identity_test_error'], .5)
        self.assertLess(row['orthogonal_test_error'], 1e-10)

    def test_zero_variance_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'zero supported variance'):
            whiten_fit(torch.ones(30, 4, dtype=torch.float64))

    def test_all_layer_pipeline(self):
        class Attention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(4, 4)
                self.k_proj = nn.Linear(4, 4)
                self.v_proj = nn.Linear(4, 4)

            def forward(self, x):
                return x + .1 * (self.q_proj(x) + self.k_proj(x) + self.v_proj(x))

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = nn.Embedding(64, 4)
                self.layers = nn.ModuleList([Attention() for _ in range(3)])
                self.fail = False

            def forward(self, input_ids, use_cache=False):
                x = self.emb(input_ids)
                for layer in self.layers:
                    x = layer(x)
                if self.fail:
                    raise RuntimeError('failure')
                return x

        model = Backbone().eval()
        weights = {i: {k: {0: (getattr(a, k+'_proj').weight.detach(),
                                getattr(a, k+'_proj').bias.detach())} for k in ('q','k','v')}
                   for i, a in enumerate(model.layers)}
        args = SimpleNamespace(analysis_device='cpu', fit_seqs=3, seqs=4, samples=8, seed=42,
                               whitening_rtol=1e-8, allow_partial_whitening=False,
                               ridge=1e-4, subspace_rank=2, save_maps=False)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            (out/'tiny').mkdir()
            ids = torch.arange(12).unsqueeze(0)
            original = model(ids)
            for s in range(4):
                capture_inputs(model, dict(enumerate(model.layers)), ids + s*12,
                               torch.arange(8), out/'tiny', s)
            self.assertTrue(torch.equal(original, model(ids)))
            analyze(args, {i: {'width':4} for i in range(3)}, weights, ['tiny'], out, out)
            rows = [json.loads(l) for l in (out/'layer_spectra.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 72)
            self.assertEqual({r['layer'] for r in rows}, {0,1,2})
            for row in rows:
                if row['control']=='whitened' and row['split']=='calibration':
                    self.assertLess(row['relative_covariance_difference'], 1e-9)
            pairs = [json.loads(l) for l in (out/'pair_spectra.jsonl').read_text().splitlines()]
            self.assertEqual(len(pairs),30)
            self.assertEqual({(r['layer_i'],r['layer_j']) for r in pairs}, {(0,1),(1,2)})
            model.fail = True
            with self.assertRaisesRegex(RuntimeError, 'failure'):
                capture_inputs(model, dict(enumerate(model.layers)), ids, torch.arange(8), out/'tiny', 9)
            self.assertTrue(all(not a.q_proj._forward_pre_hooks for a in model.layers))


if __name__ == '__main__':
    unittest.main()
