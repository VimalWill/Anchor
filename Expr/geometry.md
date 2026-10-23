# Layer geometry diagnostic

## Per-layer weight, hidden-state, and rotation decomposition

`diag_projection_controls.py` extracts raw Q/K/V weights and captures their
shared hidden inputs at every decoder layer. For each native head it compares
`x @ W.T` for actual hidden states and standard Gaussian controls, then checks
the covariance prediction `W @ Cov(x) @ W.T`. This uses covariance directly
and does not require full-rank whitening, so early layers are included.

Run on HPC from the repository root:

```bash
python Expr/diag_projection_controls.py \
  --model-kind ttt \
  --cfg Configs/ttt_ar_l2_longalpaca.yml \
  --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/nonunified_v2/ttt_at_l2_longalpaca/best_ckpt \
  --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/nonunified_v2/ttt_ar_l2_longalpaca/best_ckpt \
  --base meta-llama/Llama-3.1-8B \
  --heads 0 \
  --out /work/hdd/bgly/vwilliam/geometry/llama_anchorF_rotation_decomp
```

By default it checks head 0 at every layer. Use `--all-heads` for all native
Q/K/V heads, or `--layers 0 1` for a small run. The position cutoff and sequence
length default to 512 and 8,192. `--tokens-file` can reuse a token file made
earlier for the same tokenizer.

`W @ W.T` is the output covariance for isotropic Gaussian input. If its axes
differ from the actual output covariance axes, compare against
`W @ Cov(x) @ W.T`; close agreement with the empirical covariance shows how
the hidden covariance reshapes the projected distribution at second order.
The JSON reports covariance spectra, relative prediction errors, top-subspace
overlap, and marginal distributions of weights and hidden activations. Outputs
are JSON/JSONL summaries and `.npy` arrays only: layer Q/K/V weights and
sampled hidden states. It produces no cross-layer comparisons, CSV, PT, or
fitted-map files. This verifies a covariance-level explanation, not by itself
a causal claim about the full recurrent state.

## Original crossed-projection diagnostic

Run from the repository root in the existing CUDA evaluation environment.
Copy `Expr/diag_geometry.py` to the HPC checkout first. This is an evaluation
diagnostic: it does not train the model or modify checkpoint weights.

Example for the interleaving checkpoint (replace the checkpoint and adapter paths):

```bash
python Expr/diag_geometry.py \
  --model-kind ttt \
  --cfg Configs/ttt_ar_hybrid3_full_niah_stage3.yml \
  --ckpt /path/to/full-checkpoint \
  --base meta-llama/Llama-3.1-8B \
  --pairs 0:1 0:2 --heads 0 1 \
  --corpora pg19 wikitext \
  --seq-len 8192 --seqs 8 --fit-seqs 6 --samples 256 \
  --out geometry_ttt
```

Add `--adapter /path/to/stage2-adapter` when trained weights are separate.
For Anchor use its matching evaluation config and tokenizer. Choose layer
pairs that belong to the sharing groups of that config.

For LoLCATs, use `--model-kind lolcats`, omit `--cfg`, and supply its checkpoint
and optional adapter. Use the correct tokenizer via `--base`. Reuse
`--tokens-file geometry_ttt/tokens.pt` only with the same tokenizer identifier
and sequence settings. Use the same seed, sampled positions, pairs, and heads
for paired comparisons. A Llama token file cannot be used for Mistral.

## Measurements

- Q/K/V are raw projection outputs before RoPE, GQA repetition, and feature
  maps. Heads are native projection heads: K/V have fewer heads than Q under
  GQA. Head index 0 in Q is not a claim of equivalence to KV head 0's role.
- Hidden features are the actual shared input to the three projections,
  normally after the decoder's input normalization.
- The first six sequences calibrate means, PCA, and maps; the final two
  evaluate them. PG19 uses separate book prefixes; WikiText uses contiguous
  token blocks, which are not guaranteed to be different documents.
- Hidden maps use a common, unwhitened, calibration-only PCA basis (default
  rank 64). Raw Q/K/V maps use the full selected head dimension.
- Each pair includes five comparisons: observed `Hi Wi -> Hj Wj`, changing
  weights at fixed Hi or Hj, and changing hidden inputs at fixed Wi or Wj.
  Projection biases, if present, are included. These crossed measurements
  describe conditional effects, not an additive causal attribution.

## Outputs and interpretation

`summary.csv` reports train/evaluation errors for identity, orthogonal,
scaled-orthogonal, and ridge-linear fits. Errors use calibration-centered
targets as the denominator; the constant calibration-mean predictor has error
one when that denominator is nonzero. Mean shifts are reported separately.

`spectra.jsonl` adds activation singular values, covariance eigenvalues,
principal angles, and fitted-map singular values. Hidden spectra in this file
are in the retained PCA coordinates. `*_hidden_basis.pt` contains the pooled
full hidden spectrum and the common PCA basis. Other `.pt` files store fitted
maps, means, and polar factors. `tokens.pt` supports exactly matched reruns.
`metadata.json` records provenance and marks successful completion.

For row vectors, `Y ~= (X - mean_x) @ T + mean_y`. The saved polar factors
satisfy `T = stretch @ orthogonal`. Orthogonal maps can include reflections.
Compare held-out errors: a general linear improvement over scaled orthogonal
supports directional deformation beyond uniform scaling. Singular values
alone do not detect rotation. Polar factors on unsupported/rank-deficient
directions are not unique; inspect the recorded ranks and ridge sensitivity.

This diagnostic does not collect recurrent states or establish their causal
mechanism. It tests whether the observed geometry already exists upstream in
hidden representations and raw projections. No Gaussian assumption is needed.

Start with one pair and one head to measure throughput before expanding.
The default head selection is only head 0, not all heads. No GPU runtime is
promised. Use a fresh output directory for every run.
