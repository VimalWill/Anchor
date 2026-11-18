"""Inference driver for Nsight Compute memory profiling and dataflow tracing.

Two modes, run on the SAME checkpoint, phase, batch and length so their outputs join:

  --mode ncu    Run under `ncu` (see profiling/ncu_memory.sh). Every nn.Module and
                every TTT/attention function is wrapped in an NVTX push/pop range
                named after it, so each kernel in the report carries the module
                path that launched it. Capture is limited with cudaProfilerStart/Stop
                to the chosen phase and layers (ncu needs --profile-from-start off),
                after a warmup that has already compiled every kernel.

  --mode trace  No profiler. Runs the same phase eagerly under a TorchDispatchMode
                and records, for every aten op, which scope produced each input
                tensor. Output is a producer -> consumer edge list in LOGICAL bytes
                (tensor sizes), which is how data flows between functions. ncu's
                numbers are the PHYSICAL bytes moved through L1/shared, L2 and DRAM.
                The analyzer divides one by the other.

The trace runs eagerly with the chunked window path, because a dispatch mode cannot
see inside compiled graphs. Fused kernels therefore look different in the two
modes, but tensors crossing a module boundary are the same. Join the two at module
granularity, not op granularity.

    python profiling/profile_inference.py --mode trace --cfg CFG --ckpt CKPT --seq-len 2048 --out prof/run
    bash profiling/ncu_memory.sh prof/run --cfg CFG --ckpt CKPT --seq-len 2048
    python profiling/analyze_ncu.py prof/run
"""
import argparse
import contextlib
import functools
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------- scopes
# One stack drives both NVTX (mode ncu) and dataflow attribution (mode trace).
_STACK = []
_NVTX = False


def push(name):
    _STACK.append(name)
    if _NVTX:
        torch.cuda.nvtx.range_push(name)


def pop():
    _STACK.pop()
    if _NVTX:
        torch.cuda.nvtx.range_pop()


@contextlib.contextmanager
def scope(name):
    push(name)
    try:
        yield
    finally:
        pop()


def scope_key():
    """Innermost module path plus any function ranges opened inside it.

    e.g. 'model.layers.3.self_attn/block_causal_lact_swiglu_l2'. The phase range
    ('prefill'/'decode') is excluded; it is recorded separately.
    """
    mod, fns = None, []
    for s in _STACK:
        if s in ('prefill', 'decode'):
            continue
        if s.startswith('fn:'):
            fns.append(s[3:])
        else:
            mod, fns = s, []
    return '/'.join(([mod] if mod else ['<root>']) + fns)


def instrument_modules(model):
    """NVTX/scope push-pop around every module's forward, named by its path."""
    handles = []
    for name, mod in model.named_modules():
        name = name or '<model>'

        def pre(_m, _a, name=name):
            push(name)

        def post(_m, _a, _o):
            pop()
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    return handles


# Functions that launch the interesting kernels but are not nn.Modules.
_MODULE_FUNCS = ('block_causal_lact_swiglu', 'prenorm_block_causal_lact_swiglu',
                 'block_causal_lact_swiglu_l2', 'read_lact_swiglu_l2',
                 'sliding_window_attention', '_sdpa_sliding_window')
_METHODS = ('_decode_step', '_apply_deferred_chunk', '_ttt_features',
            '_merge_ttt', '_ttt_gate', '_align_ttt_readout')


def _wrap(fn, name):
    if getattr(fn, '_prof_wrapped', False):
        return fn

    @functools.wraps(fn)
    def w(*a, **k):
        with scope('fn:' + name):
            return fn(*a, **k)
    w._prof_wrapped = True
    return w


def instrument_functions():
    """Wrap module-level TTT/attention functions and LinearTTTAttention methods.

    The call sites look these names up as module globals or class attributes at
    call time, so rebinding them here is enough. Nothing compiled is wrapped from
    the inside: the wrappers sit outside the torch.compile boundaries.
    """
    import torch.nn.functional as F
    from LinearTTT.model.LinearizeLlama import LinearizeLlama as L
    from LinearTTT.model.LinearizeMistral import LinearizeMistral as M
    for mod in (L, M):
        for n in _MODULE_FUNCS:
            if hasattr(mod, n):
                setattr(mod, n, _wrap(getattr(mod, n), n))
        cls = getattr(mod, 'LinearTTTAttention', None)
        if cls is not None:
            for n in _METHODS:
                if hasattr(cls, n):
                    setattr(cls, n, _wrap(getattr(cls, n), n))
    F.scaled_dot_product_attention = _wrap(F.scaled_dot_product_attention, 'sdpa')


# ---------------------------------------------------------------- capture gate
class Capture:
    """cudaProfilerStart/Stop around the chosen phase, restricted to chosen layers."""

    def __init__(self, layers):
        self.layers = layers          # None = whole phase
        self.armed = False
        self.on = False
        self.cudart = torch.cuda.cudart()

    def start(self):
        if not self.on:
            self.cudart.cudaProfilerStart()
            self.on = True

    def stop(self):
        if self.on:
            self.cudart.cudaProfilerStop()
            self.on = False

    def hook_layers(self, decoder_layers):
        if self.layers is None:
            return []
        # Hooks must return None: a non-None return replaces the args/output.
        def pre(*_):
            if self.armed:
                self.start()

        def post(*_):
            if self.armed:
                self.stop()
        hs = []
        for i, layer in enumerate(decoder_layers):
            if i in self.layers:
                hs.append(layer.register_forward_pre_hook(pre))
                hs.append(layer.register_forward_hook(post))
        return hs

    @contextlib.contextmanager
    def phase(self):
        self.armed = True
        if self.layers is None:
            self.start()
        try:
            yield
        finally:
            self.stop()
            self.armed = False


def decoder_layers(model):
    base = getattr(model, 'model', model)
    base = getattr(base, 'model', base)        # PEFT wraps one level deeper
    layers = getattr(base, 'layers', None)
    if layers is None:
        raise RuntimeError('could not find model.layers')
    return layers


# ---------------------------------------------------------------- dataflow trace
class DataflowTrace(torch.utils._python_dispatch.TorchDispatchMode):
    """Producer->consumer edges between scopes, in logical tensor bytes.

    A tensor's producer is the last op that wrote its storage. Storage never
    written inside the traced region is labelled by origin: a parameter
    (param:<module>), a cache tensor carried in from the previous step
    (state:in), or the input ids (input).
    """

    def __init__(self, param_owner, device='cuda'):
        super().__init__()
        self.device = device
        self.param_owner = param_owner     # storage ptr -> owning module path
        self.producer = {}                 # storage ptr -> scope
        self.edges = defaultdict(lambda: [0, 0, None])   # (src, dst) -> bytes, n, shape
        self.ops = defaultdict(int)        # scope -> op count
        self.written = defaultdict(int)    # scope -> output bytes

    @staticmethod
    def _tensors(x):
        out = []
        torch.utils._pytree.tree_map(
            lambda t: out.append(t) if isinstance(t, torch.Tensor) else None, x)
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        dst = scope_key()
        seen = set()
        for t in self._tensors((args, kwargs)):
            if t.device.type != self.device:
                continue
            p = t.untyped_storage().data_ptr()
            if p in seen:
                continue
            seen.add(p)
            if p in self.producer:
                src = self.producer[p]
            elif p in self.param_owner:
                src = 'param:' + self.param_owner[p]
            else:
                src = 'state:in'
            if src == dst:
                continue                   # intra-scope traffic: not a function boundary
            e = self.edges[(src, dst)]
            e[0] += t.numel() * t.element_size()
            e[1] += 1
            e[2] = e[2] or list(t.shape)
        out = func(*args, **kwargs)
        self.ops[dst] += 1
        for t in self._tensors(out):
            if t.device.type == self.device:
                self.producer[t.untyped_storage().data_ptr()] = dst
                self.written[dst] += t.numel() * t.element_size()
        return out

    def dump(self, path, phase):
        rows = [dict(phase=phase, src=s, dst=d, bytes=b, tensors=n, shape=sh)
                for (s, d), (b, n, sh) in sorted(self.edges.items(),
                                                 key=lambda kv: -kv[1][0])]
        with open(path, 'w') as f:
            json.dump({'phase': phase, 'edges': rows,
                       'ops': dict(self.ops), 'written': dict(self.written)}, f)
        return rows


# ---------------------------------------------------------------- driver
def parse_layers(spec, n):
    if spec in (None, 'all'):
        return None
    out = set()
    for tok in spec.split(','):
        if '-' in tok:
            a, b = tok.split('-')
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(tok) if int(tok) >= 0 else n + int(tok))
    return out


def load(a):
    import LinearTTT  # noqa: F401
    from measure_flops import load_benchmark_config, set_window_backend
    if a.baseline:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, device_map={'': 0}).eval()
    from Training.train import build_model_config
    from eval import load_model
    set_window_backend(a.mode == 'trace')
    cfg = load_benchmark_config(a.cfg, a.ckpt, a.seq_len + a.decode_warmup + a.decode_steps + 1)
    return load_model(a.ckpt, build_model_config(cfg), a.adapter, verbose=False).eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['ncu', 'trace'], required=True)
    ap.add_argument('--cfg')
    ap.add_argument('--ckpt')
    ap.add_argument('--adapter')
    ap.add_argument('--baseline', action='store_true', help='profile the unmodified teacher')
    ap.add_argument('--base', default='meta-llama/Llama-3.1-8B')
    ap.add_argument('--seq-len', type=int, default=2048)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--phase', choices=['prefill', 'decode', 'both'], default='both')
    ap.add_argument('--decode-warmup', type=int, default=4,
                    help='unprofiled decode steps before the measured ones')
    ap.add_argument('--decode-steps', type=int, default=1,
                    help='measured decode steps; each launches every kernel again')
    ap.add_argument('--layers', default='0,-1',
                    help="decoder layers to capture: 'all', or e.g. '0,3,28-31,-1'. "
                         "Embedding and lm_head are captured only with 'all'. Every "
                         'captured kernel is replayed once per metric pass, so '
                         "'all' at prefill is slow")
    ap.add_argument('--warmup', type=int, default=2, help='full requests before capture (ncu mode)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--out', required=True, help='output prefix, e.g. prof/run')
    a = ap.parse_args()
    if not a.baseline and not (a.cfg and a.ckpt):
        ap.error('--cfg and --ckpt are required unless --baseline')

    global _NVTX
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch._dynamo.config.cache_size_limit = 256
    torch.manual_seed(a.seed)

    model = load(a)
    for p in model.parameters():
        p.requires_grad_(False)
    from measure_flops import inference_kwargs, checked_cache
    kwargs = inference_kwargs(model)
    dev = model.device
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    ids = torch.randint(model.config.vocab_size, (a.batch, a.seq_len), device=dev, generator=gen)
    layers = decoder_layers(model)
    sel = parse_layers(a.layers, len(layers))
    phases = ['prefill', 'decode'] if a.phase == 'both' else [a.phase]

    instrument_functions()
    instrument_modules(model)

    meta = dict(mode=a.mode, cfg=a.cfg, ckpt=a.ckpt, baseline=a.baseline,
                seq_len=a.seq_len, batch=a.batch, phases=phases,
                layers=sorted(sel) if sel is not None else 'all',
                n_layers=len(layers), decode_warmup=a.decode_warmup,
                decode_steps=a.decode_steps, gpu=torch.cuda.get_device_name(dev),
                ttt_layer_indices=getattr(model.config, 'ttt_layer_indices', None),
                ttt_share_groups=getattr(model.config, 'ttt_share_groups', None),
                window_size=getattr(model.config, 'window_size', None),
                lact_chunk_size=getattr(model.config, 'lact_chunk_size', None))

    def request(run_phase):
        """One fresh request. run_phase(name, fn) wraps each measured part."""
        out = run_phase('prefill', lambda: model(input_ids=ids, use_cache=True, **kwargs))
        cache = checked_cache(out, ids.shape[1])
        tok = out.logits[:, -1].argmax(-1, keepdim=True)
        del out
        for step in range(a.decode_warmup + a.decode_steps):
            f = lambda: model(input_ids=tok, past_key_values=cache, use_cache=True, **kwargs)
            out = run_phase('decode', f) if step >= a.decode_warmup else f()
            cache = checked_cache(out, ids.shape[1] + step + 1)
            tok = out.logits[:, -1].argmax(-1, keepdim=True)
            del out

    with torch.no_grad():
        if a.mode == 'ncu':
            for _ in range(a.warmup):              # compiles every shape we will capture
                request(lambda _n, f: f())
            torch.cuda.synchronize()
            cap = Capture(sel)
            cap.hook_layers(layers)
            _NVTX = True

            def run_phase(name, f):
                if name not in phases:
                    return f()
                with scope(name), cap.phase():
                    return f()
            request(run_phase)
            torch.cuda.synchronize()
            meta_path = f'{a.out}.ncu-meta.json'
        else:
            owner = {}
            for mname, m in model.named_modules():
                for p in list(m.parameters(recurse=False)) + list(m.buffers(recurse=False)):
                    owner[p.untyped_storage().data_ptr()] = mname
            traces = {}

            def run_phase(name, f):
                if name not in phases:
                    return f()
                tr = DataflowTrace(owner)
                with torch._dynamo.config.patch(disable=True), scope(name), tr:
                    out = f()
                traces.setdefault(name, []).append(tr)
                return out
            request(run_phase)
            for name, trs in traces.items():
                rows = trs[-1].dump(f'{a.out}.dataflow_{name}.json', name)
                print(f'{name}: {len(rows)} cross-scope edges, '
                      f'{sum(r["bytes"] for r in rows) / 2**20:,.1f} MiB logical '
                      f'-> {a.out}.dataflow_{name}.json')
            meta_path = f'{a.out}.trace-meta.json'
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'wrote {meta_path}')


if __name__ == '__main__':
    main()
