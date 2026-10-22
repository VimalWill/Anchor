# Layer geometry diagnostic

## Every-layer whitening and Gaussian controls

Use `diag_projection_controls.py` for the identity-covariance attribution
experiment. It supports TTT and LoLCATs through the same loaders as the original
diagnostic. It selects **every decoder layer by default**, including layers
whose TTT branch is disabled, and compares every adjacent layer pair. It probes
raw Q/K/V before RoPE and before SiLU/feature mapping.

For the existing Llama Anchor-F checkpoint, run from the HPC repository root:

```bash
python Expr/diag_projection_controls.py \
  --model-kind ttt \
  --cfg Configs/ttt_ar_l2_longalpaca.yml \
  --ckpt /work/nvme/bgly/vwilliam/ttt-checkpoints/nonunified_v2/ttt_at_l2_longalpaca/best_ckpt \
  --adapter /work/nvme/bgly/vwilliam/ttt-checkpoints/nonunified_v2/ttt_ar_l2_longalpaca/best_ckpt \
  --base meta-llama/Llama-3.1-8B \
  --tokens-file /work/hdd/bgly/vwilliam/geometry/llama_anchorF_8192/tokens.pt \
  --heads 0 \
  --out /work/hdd/bgly/vwilliam/geometry/llama_anchorF_controls_all_layers
```

Sync `Expr/diag_projection_controls.py`, `Expr/diag_geometry.py`, and
`Expr/diag_rank.py` to the HPC. The old token file can be reused: the new probe
samples more positions from those same sequences. A new output directory is
required. Use the appropriate config, checkpoint, adapter, and tokenizer for
other architectures; do not reuse tokens across different tokenizers.

The command measures head 0 at **all 32 layers**. To cover all native heads,
replace `--heads 0` with `--all-heads`: 32 Q heads and 8 K/V heads for this
checkpoint, without GQA duplication. This substantially increases analysis
work. No GPU runtime has been measured. `--layers 0 1` can restrict a smoke run.

There are three measured inputs and two analytic references:

| Label | Input / covariance reference |
|---|---|
| `actual` | Captured real projection input, recomputed with merged weights in float64 |
| `whitened` | Real input transformed with per-layer symmetric ZCA |
| `gaussian` | The same standard Gaussian samples at every layer, with separate calibration/evaluation rows |
| `identity_exact` | Exact population output covariance `W W.T` |
| `supported_exact` | `W P W.T`, where P projects onto the supported calibration hidden subspace |

Defaults use 8 sequences, 1,024 sampled positions each, and the first 6
sequences for calibration: **6,144 calibration samples** versus hidden width
4,096. The last 2 sequences evaluate fits and apply the already fitted ZCA;
they are never used to calibrate whitening. Their covariance can differ from
identity due to sampling and distribution shift. Gaussian output covariance
also has finite-sample error. Biases affect the mean but not centered covariance.

Full whitening is required by default, with a relative eigenvalue threshold of
1e-8. If it is rank deficient, increase samples/sequences or investigate the
reported spectrum. `--allow-partial-whitening` explicitly permits projection
onto the supported subspace, but that is **not** an identity-covariance test.
The output marks `full_identity: false` and uses `supported_exact` as the
whitened baseline. Do not lower the threshold solely to force a full rank.

The 512-token position cutoff and 8,192 sequence length carry over from the
previous experiment. The sequence split is by PG19 book and by contiguous
WikiText blocks; WikiText blocks are not guaranteed to be separate documents.

Outputs:

- `whitening_summary.csv`: rank, conditioning, and actual calibration
  covariance error. Inspect this first.
- `layer_summary.csv` / `layer_spectra.jsonl`: all-layer covariance traces,
  spectra, effective ranks, and deviation from each analytic baseline.
- `pair_summary.csv` / `pair_spectra.jsonl`: every adjacent pair, orthogonal,
  scaled-orthogonal, and general linear held-out fits for all three inputs;
  also covariance changes under exact identity inputs.
- `*_covariances.pt`: full output covariance matrices for inspecting axes or
  deriving additional subspace comparisons. Add `--save-maps` for fitted maps
  and their polar decompositions.
- `metadata.json`: definitions, checkpoints, code/token hashes, selected layers
  and heads, and the completion flag.

For 32 layers, 8 sequences, 1,024 samples, and hidden width 4,096, temporary
hidden-input files require about **4 GiB per corpus** (8 GiB for both). They
are stored under the output directory and removed when the process exits
normally or raises a Python exception; abrupt termination can leave them.
Activations are analyzed one layer at a time after releasing the model.

Interpret covariance changes and paired fits together. Identity input removes
the real input covariance but does not remove the weights' differing subspaces.
If a pattern persists, weights are sufficient to generate it in this control;
if it disappears, hidden covariance/alignment mattered. This is not a causal
percentage decomposition or proof of the mechanism in the recurrent state.
Whitened real inputs need not be Gaussian. Rotations of isotropic inputs are
invisible to population covariance; shared paired Gaussian outputs provide
the complementary alignment measurement.

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
