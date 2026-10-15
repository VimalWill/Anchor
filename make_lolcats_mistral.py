"""Emit a LoLCATs model directory for a Mistral backbone.

No modeling code is ported. LoLCATs subclasses the Llama stack, but Llama and
Mistral have identical state-dict key sets and tensor shapes at matched
dimensions, and LolcatsAttention replaces self_attn outright -- so Mistral's
one structural difference, its native 4096 sliding window, is discarded along
with the attention it belonged to. Everything that differs is a config scalar.

Scalars are read from the real Mistral config rather than typed in here. The
linear-attention fields are copied from Liger's own lolcats_base config so the
Mistral run is comparable to their Llama one.
"""
import argparse
import json
from pathlib import Path

LOLCATS_FIELDS = {
    'attn': None,
    'attn_mode': 'fused_chunk',
    'expand_k': 1,
    'expand_v': 1,
    'feature_map': 'lolcats_hedgehog',
    'tie_feature_map_qk': False,
    'norm_q': False,
    'norm_k': False,
    'norm_feature_map': False,
    'elementwise_affine': True,
    'norm_eps': 1e-06,
    'hidden_ratio': 4,
    'fuse_cross_entropy': True,
}
# Carried from the backbone; everything else in LolcatsConfig keeps its default.
BACKBONE_FIELDS = (
    'vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
    'num_attention_heads', 'num_key_value_heads', 'head_dim', 'hidden_act',
    'max_position_embeddings', 'initializer_range', 'rms_norm_eps',
    'rope_theta', 'rope_scaling', 'attention_bias', 'attention_dropout',
    'mlp_bias', 'bos_token_id', 'eos_token_id', 'pad_token_id',
    'tie_word_embeddings', 'pretraining_tp',
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base', default='mistralai/Mistral-7B-v0.1')
    ap.add_argument('--out', required=True, help='new directory to write config.json into')
    ap.add_argument('--tokenizer', action='store_true',
                    help='also save the backbone tokenizer, so --out loads standalone')
    args = ap.parse_args()

    from transformers import AutoConfig
    source = AutoConfig.from_pretrained(args.base).to_dict()
    if source.get('sliding_window'):
        print(f"note: dropping the backbone's sliding_window={source['sliding_window']}; "
              'LolcatsAttention replaces self_attn and uses its own window of 64')

    config = {k: source[k] for k in BACKBONE_FIELDS if k in source}
    config.setdefault('head_dim', source['hidden_size'] // source['num_attention_heads'])
    config.update(LOLCATS_FIELDS)
    config['model_type'] = 'lolcats'
    config['architectures'] = ['LolcatsModelForCausalLM']
    config['torch_dtype'] = 'bfloat16'

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    (out / 'config.json').write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    if args.tokenizer:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(args.base).save_pretrained(out)
    print(f'wrote {out / "config.json"} from {args.base}')
    for key in ('vocab_size', 'intermediate_size', 'rope_theta', 'max_position_embeddings'):
        print(f'  {key}: {config[key]}')


if __name__ == '__main__':
    main()
