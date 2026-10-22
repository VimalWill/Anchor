# Anchor

**Augmented Cross-Layer Recurrent State for Long-Context Retrieval**

Anchor linearizes a pretrained Transformer (Llama-3.1-8B, Mistral-7B) into a hybrid of
sliding-window attention (SWA, window 512) and a Test-Time Training (TTT) memory. Long-context
recall turns out to live in only a few boundary layers (`0, 1, 28–31`), and those layers hold the
same information expressed in different bases. So Anchor keeps **one shared TTT state per anchor
group**. One layer writes it, and every other layer in the group reads it through a learned,
identity-initialized read map `χ_ℓ`.

| Config     | TTT memories                          | State (32K) |
|------------|---------------------------------------|-------------|
| Anchor-F   | one per layer                         | 201 MB      |
| Anchor-I   | shared anchors + interior (training)  | 176 MB      |
| **Anchor** | shared anchors only (deployment)      | **12.6 MB** |

## Setup

```bash
git clone --recursive https://github.com/VimalWill/Anchor.git && cd Anchor
pip install -r requirements.txt                        
export PYTHONPATH=.:$PYTHONPATH
```

> tested on NVIDIA GH200 GPU

## Training

Training follows the two-stage recipe,
1. **Stage 1** (attention transfer, `MSE + CE`) fits the
TTT and gate parameters against the frozen teacher. 
2. **Stage 2** (rank-8 LoRA on q/k/v, CE) adapts
on long context. Checkpoints go to `$TTT_CKPT_DIR/<config-name>/best_ckpt` (default `checkpoints/`).

    ```bash
    # Llama-3.1-8B 
    python run.py --cfg Configs/ttt_at_anchor_gate.yml     
    python run.py --cfg Configs/ttt_ar_unified_longalpaca.yml 

    # Mistral-7B
    python run.py --cfg Configs/ttt_at_mistral_anchor.yml
    python run.py --cfg Configs/ttt_ar_mistral_anchor.yml
    ```


3. **Hybrid LLM Architecture (3:1).** Three Anchor layers share one memory, and every fourth
layer uses full attention. Stage 2 trains on passkey data mixed with synthetic needle-in-a-haystack
data:

    ```bash
    python run.py --cfg Configs/ttt_at_hybrid3_full.yml
    TTT_INIT=checkpoints/ttt_at_hybrid3_full/best_ckpt bash scripts/train_stage2_mixed.sh
    ```

## Evaluation

`eval.py` wraps `lm_eval`. Pass the **stage-1** checkpoint as `--ckpt` and the stage-2 output as
`--adapter`. Never pass a base-model `--ckpt`. To get the deployed **Anchor** model (interior
memories removed), evaluate the same checkpoints under the deploy config.

```bash
CK=checkpoints
ARGS="--ckpt $CK/ttt_at_anchor_gate/best_ckpt --adapter $CK/ttt_ar_unified_longalpaca/best_ckpt"

python eval.py --cfg Configs/ttt_deploy_anchor.yml $ARGS --suite commonsense --out eval/anchor_cs
python eval.py --cfg Configs/ttt_deploy_anchor.yml $ARGS --suite recall      --out eval/anchor_recall
python eval.py --cfg Configs/ttt_deploy_anchor.yml $ARGS --suite mmlu        --out eval/anchor_mmlu

# Anchor-I: same checkpoints, training config
python eval.py --cfg Configs/ttt_ar_unified_longalpaca.yml $ARGS --suite recall --out eval/anchorI_recall

# Teacher reference row
python eval.py --baseline --suite commonsense recall --out eval/llama
```

To evaluate Mistral, use `Configs/ttt_deploy_mistral_anchor.yml`. For the Mistral teacher row, run
`--baseline --base mistralai/Mistral-7B-v0.1`.

| Flag | Purpose |
|------|---------|
| `--suite {commonsense,recall,mmlu}` | Runs the benchmarks in Table 2 and Figure 6 |
| `--tasks ruler --ruler-lengths 4096 8192 16384 --seq-len 16896` | Runs RULER (Table 4). `--seq-len` is required, or inputs are truncated to 8K |
| `--retrieval --branch both --seqs 32` | Runs the per-layer 

> Representation analyses (CKA and cosine for Figures 5 and 9, read-map rank for Figure 8, FLOPs, and latency) live in `Expr/`.


