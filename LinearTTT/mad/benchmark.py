"""MAD benchmark for ANCHOR-I / ANCHOR, the 3:1 interleave, and a Transformer.

Runs MAD's own protocol -- make_benchmark_mad_configs() and train() from
mad-lab, so the task grid, the lr x wd sweep, the epochs and the final-model
test accuracy are exactly what compute_model_mad_scores() aggregates.

Four mixers, each followed by SwiGLU, width 128:

  anchor-i     [H* H  H  H*]  anchors 0 and 3 share one memory (0 writes)
  anchor-f     [H  H  H  H ]  every layer its own memory, no sharing
  anchor       the anchor-i model with the interior memories switched off at
               evaluation, as the deploy config drops them; not trained
  interleave   [H* H* H* S ]  three hybrids share one memory, then SWA only
  transformer  [A  A  A  A ]  MAD's stock mh-attention

H = SWA + TTT hybrid, S = SWA only. MAD sequences are 32-1024 tokens, so the
window is scaled down from the 512/8192 of the 32-layer models: SWA 32, TTT
chunk 16. Otherwise a window would cover most sequences and the memory would
not be tested.

--single-sweep keeps every task's full difficulty grid but only the
lr=5e-4, wd=0 sweep point (66 runs per architecture instead of 396). Scores
are then the mean test accuracy over each task's difficulty settings, not
MAD's best-of-sweep, so they compare these architectures with each other,
not with published MAD tables.

Usage (from the repo root):
  python -m LinearTTT.mad.benchmark prepare --data DIR
  python -m LinearTTT.mad.benchmark run --arch anchor-i --data DIR --logs DIR \
      [--tasks in-context-recall] [--pilot] [--single-sweep] [--shard 0 --shards 4]
  python -m LinearTTT.mad.benchmark score --logs DIR [--single-sweep --data DIR]
"""
import argparse
import copy
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'third_party' / 'mad-lab'))

# train() ends with trainer.validate(), which reloads the run's best checkpoint;
# torch >= 2.6 refuses it by default because it pickles MADConfig. These are
# checkpoints this same process just wrote.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

from LinearTTT.mad.register import link_shared_memories, register_ttt  # noqa: E402

register_ttt()  # before any mad.model import, so attention is the SDPA one

DIM, HEADS, ROTARY = 128, 4, 8
WINDOW, CHUNK = 32, 16
ARCHS = ('anchor-i', 'anchor-f', 'interleave', 'transformer')
SCORED = ('anchor-i', 'anchor', 'anchor-f', 'interleave', 'transformer')


def build(arch, vocab_size, max_length):
    from mad.model.layers import SwiGLU
    from mad.registry import layer_registry, model_registry
    from LinearTTT.mad.attention import SDPAAttention
    from LinearTTT.mad.hybrid import TTTHybridBlock

    base = dict(dim=DIM, max_length=max_length)
    hybrid = dict(base, num_heads=HEADS, chunk_size=CHUNK, window_size=WINDOW,
                  rotary_emb_dim=ROTARY)
    swa = dict(base, n_heads=HEADS, rotary_emb_dim=ROTARY, window_size=(WINDOW, 0))
    if arch == 'transformer':
        attention = layer_registry['mh-attention']
        import yaml
        cfg = yaml.safe_load(open(ROOT / 'third_party' / 'mad-lab' / attention['cfg']))
        mixers = [(attention['module'], dict(cfg, **base))] * 4
        groups = []
    elif arch == 'anchor-i':
        mixers = [(TTTHybridBlock, hybrid)] * 4
        groups = [[0, 3]]
    elif arch == 'anchor-f':
        mixers = [(TTTHybridBlock, hybrid)] * 4
        groups = []
    elif arch == 'interleave':
        mixers = [(TTTHybridBlock, hybrid)] * 3 + [(SDPAAttention, swa)]
        groups = [[0, 1, 2]]
    else:
        raise ValueError(arch)
    layers, configs = [], []
    for module, cfg in mixers:
        layers += [module, SwiGLU]
        configs += [dict(cfg), dict(base)]
    # The registry class keeps the TTT gate/retention init that MAD would reset.
    model = model_registry['language-model'](
        vocab_size=vocab_size, layers=layers, layer_cfgs=configs, dim=DIM, max_length=max_length)
    if groups:
        link_shared_memories(model, groups)  # before the optimizer exists
    return model


def deploy(model):
    """ANCHOR from ANCHOR-I: interior memories off, anchors untouched."""
    from LinearTTT.mad.hybrid import TTTHybridBlock
    blocks = [m for m in model.modules() if isinstance(m, TTTHybridBlock)]
    assert len(blocks) == 4, 'expected an anchor-i model'
    for block in blocks[1:3]:
        block.memory_enabled = False
    return model


def configs(data, tasks=None, pilot=False, single_sweep=False):
    from mad.configs import make_benchmark_mad_configs
    out = make_benchmark_mad_configs(data_path=data, precision='bf16', persistent_workers=False)
    out = [c for c in out if tasks is None or c.task in tasks]
    if single_sweep:
        out = [c for c in out if c.lr == 5e-4 and c.weight_decay == 0.0]
    if pilot:  # each task's baseline setting at one sweep point; the sweep reuses these
        first = {}
        for c in out:
            if c.lr == 5e-4 and c.weight_decay == 0.0:
                first.setdefault(c.task, c)  # the baseline is the first of its sweep block
        out = list(first.values())
    return out


def evaluate(model, mad_config):
    """The same final validate() train() runs, for a model it did not train."""
    import pandas as pd
    import pytorch_lightning as pl
    import torch
    from torch.utils.data import DataLoader
    from mad.data import generate_data
    from mad.model.pl_model_wrapper import PLModelWrap

    data = generate_data(
        instance_fn=mad_config.instance_fn, instance_fn_kwargs=mad_config.instance_fn_kwargs,
        train_data_path=mad_config.train_dataset_path, test_data_path=mad_config.test_dataset_path,
        num_train_examples=mad_config.num_train_examples,
        num_test_examples=mad_config.num_test_examples, num_workers=0)
    trainer = pl.Trainer(accelerator='cuda' if torch.cuda.is_available() else 'cpu', devices=1,
                         precision=mad_config.precision, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    wrapped = PLModelWrap(model=model, mad_config=mad_config)
    row = {}
    for split in ('train', 'test'):
        loader = DataLoader(data[split], batch_size=mad_config.batch_size, shuffle=False)
        r = trainer.validate(wrapped, dataloaders=loader, verbose=False)[0]
        row.update({f'{split}_acc': r['test/Accuracy_epoch'], f'{split}_ppl': r['test/Perplexity_epoch'],
                    f'{split}_loss': r['test/Loss_epoch']})
    return pd.DataFrame(row, index=[0])


def best_checkpoint_model(arch, mad_config, log_path):
    """The weights MAD scores: train() validates its best-perplexity checkpoint."""
    import torch
    model = build(arch, mad_config.vocab_size, mad_config.seq_len)
    state = torch.load(os.path.join(log_path, 'checkpoints', 'best.ckpt'),
                       map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict({k.removeprefix('model.'): v for k, v in state.items()})
    return model


def run(args):
    import torch
    from mad.paths import make_log_path
    from train import train

    todo = configs(args.data, args.tasks, args.pilot, args.single_sweep)[args.shard::args.shards]
    print(f'{args.arch}: {len(todo)} runs in shard {args.shard}/{args.shards}', flush=True)
    for i, mad_config in enumerate(todo):
        start = time.time()
        log_path = make_log_path(base_path=args.logs, mad_config=mad_config, model_id=args.arch)
        model = build(args.arch, mad_config.vocab_size, mad_config.seq_len)
        result = train(model=model, mad_config=mad_config, log_path=log_path,
                       log_to_csv=True, save_checkpoints=True)
        if args.arch == 'anchor-i':
            deploy_path = make_log_path(base_path=args.logs, mad_config=mad_config, model_id='anchor')
            if not os.path.exists(os.path.join(deploy_path, 'results.csv')):
                # Same weights train() scored for anchor-i, also on resume,
                # where train() returns cached results without a model.
                trained = best_checkpoint_model('anchor-i', mad_config, log_path)
                trained.to('cuda' if torch.cuda.is_available() else 'cpu')
                os.makedirs(deploy_path, exist_ok=True)
                evaluate(deploy(copy.deepcopy(trained)), mad_config).to_csv(
                    os.path.join(deploy_path, 'results.csv'), index=False)
        print(f'[{i + 1}/{len(todo)}] {mad_config.task} vocab={mad_config.vocab_size} '
              f'len={mad_config.seq_len} n={mad_config.num_train_examples} lr={mad_config.lr} '
              f'wd={mad_config.weight_decay} test_acc={float(result["test_acc"][0]):.4f} '
              f'{time.time() - start:.0f}s', flush=True)


def _loadable(c):
    import numpy as np
    for path, n in ((c.train_dataset_path, c.num_train_examples), (c.test_dataset_path, c.num_test_examples)):
        try:
            x = np.load(os.path.join(path, 'inputs.npy'))
            y = np.load(os.path.join(path, 'targets.npy'))
            if len(x) != n or len(y) != n:
                return False
        except Exception:
            return False
    return True


def _generate(c, gen_workers=0):
    """One dataset, from scratch if any split is missing or unreadable."""
    import shutil
    from mad.data import generate_data
    if _loadable(c):
        return c.train_dataset_path, 'kept'
    for path in (c.train_dataset_path, c.test_dataset_path):
        shutil.rmtree(path, ignore_errors=True)
    generate_data(instance_fn=c.instance_fn, instance_fn_kwargs=c.instance_fn_kwargs,
                  train_data_path=c.train_dataset_path, test_data_path=c.test_dataset_path,
                  num_train_examples=c.num_train_examples,
                  num_test_examples=c.num_test_examples, num_workers=gen_workers)
    return c.train_dataset_path, 'generated' if _loadable(c) else 'FAILED'


def prepare(args):
    """Generate every dataset once, in parallel; each worker owns distinct paths.

    Anything missing, partial or unreadable is regenerated, so a run that was
    interrupted mid-write (or two writers that raced) leaves nothing behind.
    """
    from multiprocessing import Pool
    unique = {}
    for c in configs(args.data, args.tasks, args.pilot, args.single_sweep):
        unique.setdefault(c.train_dataset_path, c)
    if args.workers > 1:      # many datasets at once, one core each
        with Pool(args.workers) as pool:
            status = pool.map(_generate, list(unique.values()), chunksize=1)
    else:                     # one dataset at a time, split across --gen-workers
        status = [_generate(c, args.gen_workers) for c in unique.values()]
    counts = {s: sum(1 for _, x in status if x == s) for s in ('kept', 'generated', 'FAILED')}
    print(f'{len(unique)} datasets in {args.data}: {counts}')
    if counts['FAILED']:
        raise SystemExit('some datasets could not be generated')


def score_single_sweep(args):
    """Mean final test accuracy per task over its difficulty settings, one sweep point."""
    import pandas as pd
    from mad.paths import make_log_path
    todo = configs(args.data, None, args.pilot, not args.pilot)
    rows, missing = {}, {}
    for model_id in SCORED:
        acc = {}
        for c in todo:
            path = os.path.join(make_log_path(base_path=args.logs, mad_config=c, model_id=model_id), 'results.csv')
            if os.path.exists(path):
                acc.setdefault(c.task, []).append(float(pd.read_csv(path)['test_acc'][0]))
            else:
                missing[model_id] = missing.get(model_id, 0) + 1
        if acc:
            rows[model_id] = {task: 100 * sum(v) / len(v) for task, v in acc.items()}
    table = pd.DataFrame(rows).T
    table['avg'] = table.mean(axis=1)
    print(table.round(1).to_string())
    for model_id, n in missing.items():
        print(f'{model_id}: {n}/{len(todo)} settings not finished yet')
    table.to_csv(os.path.join(args.logs, 'mad_scores_single_sweep.csv'))


def score(args):
    if args.single_sweep or args.pilot:
        return score_single_sweep(args)
    import pandas as pd
    from mad.analysis import compute_model_mad_scores
    rows = {}
    for model_id in SCORED:
        try:
            rows[model_id] = compute_model_mad_scores(model_id=model_id, logs_path=args.logs,
                                                      task=args.task)
        except AssertionError as missing:
            print(f'{model_id}: incomplete ({str(missing)[-120:]})')
    table = pd.DataFrame(rows).T * 100
    table['avg'] = table.mean(axis=1)
    print(table.round(1).to_string())
    table.to_csv(os.path.join(args.logs, 'mad_scores.csv'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare'); p.add_argument('--data', required=True); p.add_argument('--tasks', nargs='+')
    p.add_argument('--pilot', action='store_true')
    p.add_argument('--single-sweep', action='store_true')
    p.add_argument('--workers', type=int, default=1, help='parallel dataset generators')
    p.add_argument('--gen-workers', type=int, default=0, help='MAD workers within one dataset (with --workers 1)')
    r = sub.add_parser('run')
    r.add_argument('--arch', choices=ARCHS, required=True)
    r.add_argument('--data', required=True); r.add_argument('--logs', required=True)
    r.add_argument('--tasks', nargs='+')
    r.add_argument('--pilot', action='store_true', help='one baseline run per task, for timing')
    r.add_argument('--single-sweep', action='store_true', help='full difficulty grid, lr=5e-4 wd=0 only')
    r.add_argument('--shard', type=int, default=0); r.add_argument('--shards', type=int, default=1)
    s = sub.add_parser('score'); s.add_argument('--logs', required=True); s.add_argument('--task', default='all')
    s.add_argument('--single-sweep', action='store_true'); s.add_argument('--data')
    s.add_argument('--pilot', action='store_true', help='score the baseline setting of each task only')
    args = ap.parse_args()
    for key in ('data', 'logs'):
        if getattr(args, key, None):
            setattr(args, key, os.path.abspath(getattr(args, key)))
    # mad-lab opens its task configs by relative path ('configs/tasks/...')
    os.chdir(ROOT / 'third_party' / 'mad-lab')
    {'prepare': prepare, 'run': run, 'score': score}[args.cmd](args)


if __name__ == '__main__':
    main()
