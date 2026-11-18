"""Join ncu memory metrics (physical bytes) with the dataflow trace (logical bytes).

    python profiling/analyze_ncu.py prof/run [--depth 1]

Reads, for prefix P:
  P.all.ncu-rep / P.none.ncu-rep   from ncu_memory.sh (either or both)
  P.dataflow_<phase>.json          from profile_inference.py --mode trace (optional)
Writes:
  P.kernels.csv     one row per profiled kernel: scope, phase, bytes at each level
  P.scopes.csv      per (phase, scope): summed ncu bytes + logical inflow from the trace
  P.flow_<phase>.dot  scope-level dataflow graph, edges in MiB, nodes with ncu DRAM/time

Columns, per kernel or summed per scope (bytes unless noted):
  dram_rd/dram_wr    HBM <-> L2
  l2_from_l1_rd/wr   L1 <-> L2 requests (lts__t_sectors_srcunit_tex * 32)
  l2_to_l1           L2 -> L1/shared fill bytes over the crossbar
  l1_glb_ld/st       global loads/stores seen by L1
  smem_ld/st         shared-memory bytes moved by SASS ld/st
  smem_conflict_ld/st  bank-conflict wavefronts / total wavefronts (0 = conflict-free)
  l2_hit, l1_hit     percent
  logical_in         tensor bytes the scope consumed from OTHER scopes (trace)
  param_in           of which, weights/buffers
  dram_per_logical   dram_rd / logical_in: < 1 means inputs were served from L2
  warm_saved         dram_rd(cold) - dram_rd(warm): bytes found in L2 that earlier
                     kernels left there. This is the inter-function L2 flow.
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil
import sys
from collections import defaultdict

PHASES = ('prefill', 'decode')
M = {
    'time_ns': 'gpu__time_duration.sum',
    'dram_rd': 'dram__bytes_read.sum',
    'dram_wr': 'dram__bytes_write.sum',
    'l2_from_l1_rd': 'lts__t_sectors_srcunit_tex_op_read.sum',
    'l2_from_l1_wr': 'lts__t_sectors_srcunit_tex_op_write.sum',
    'l2_to_l1': 'l1tex__m_xbar2l1tex_read_bytes.sum',
    'l1_to_l2': 'l1tex__m_l1tex2xbar_write_bytes.sum',
    'l1_glb_ld': 'l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum',
    'l1_glb_st': 'l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum',
    'smem_ld': 'sm__sass_data_bytes_mem_shared_op_ld.sum',
    'smem_st': 'sm__sass_data_bytes_mem_shared_op_st.sum',
    'smem_wf_ld': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum',
    'smem_wf_st': 'l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum',
    'smem_bc_ld': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum',
    'smem_bc_st': 'l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum',
    'l2_hit': 'lts__t_sector_hit_rate.pct',
    'l1_hit': 'l1tex__t_sector_hit_rate.pct',
}
SECTOR = 32
SUMMED = [k for k in M if k not in ('l2_hit', 'l1_hit')]


def import_ncu_report():
    cands = [os.environ.get('NCU_PYTHON', '')]
    exe = shutil.which('ncu')
    if exe:
        d = os.path.dirname(os.path.realpath(exe))
        cands += [os.path.join(d, 'extras', 'python'),
                  os.path.join(d, '..', 'extras', 'python')]
        cands += glob.glob(os.path.join(d, '..', '..', 'nsight-compute*', 'extras', 'python'))
    for c in cands:
        if c and os.path.isfile(os.path.join(c, 'ncu_report.py')):
            sys.path.insert(0, c)
            import ncu_report
            return ncu_report
    raise SystemExit('ncu_report.py not found. Set NCU_PYTHON=<nsight-compute>/extras/python')


def split_stack(ranges):
    """NVTX push/pop stack -> (phase, scope key), same rule as profile_inference.scope_key."""
    phase, mod, fns = None, None, []
    for s in ranges:
        if s in PHASES:
            phase = s
        elif s.startswith('fn:'):
            fns.append(s[3:])
        else:
            mod, fns = s, []
    return phase, '/'.join(([mod] if mod else ['<root>']) + fns)


def collapse(key, depth):
    """Module path to a readable scope: depth 0 = layer, 1 = layer child, 2 = grandchild.

    Function ranges (the TTT update, window attention, ...) are kept at depth >= 1,
    because they split self_attn into its two branches.
    """
    if key.startswith('param:'):
        return 'param:' + collapse(key[6:], depth)
    if key in ('state:in', '<root>'):
        return key
    mod, *fns = key.split('/')
    parts = mod.split('.')
    if 'layers' in parts:
        i = parts.index('layers')
        if i + 1 < len(parts):
            name = f'L{parts[i + 1]}'
            rest = parts[i + 2:i + 2 + depth]
            # PEFT: lora wrappers add base_layer/lora_A levels under the real module
            rest = [r for r in rest if r not in ('base_layer', 'default')]
            name = '.'.join([name] + rest)
        else:
            name = 'layers'
    else:
        tail = [p for p in parts if p not in ('<model>', 'base_model', 'model', '')]
        name = tail[-1] if tail else '<model>'
    if depth >= 1 and fns:
        name += '/' + fns[0]
    return name


def load_ncu(path, ncu_report, depth):
    ctx = ncu_report.load_report(path)
    rows = []
    for ri in range(ctx.num_ranges()):
        rng = ctx.range_by_idx(ri)
        for ai in range(rng.num_actions()):
            act = rng.action_by_idx(ai)
            stack = []
            nv = act.nvtx_state()
            if nv is not None:
                for did in nv.domains():
                    stack += list(nv.domain_by_id(did).push_pop_ranges())
            phase, key = split_stack(stack)
            r = {'kernel': act.name(), 'phase': phase or '?', 'key': key,
                 'scope': collapse(key, depth)}
            for k, name in M.items():
                m = act.metric_by_name(name)
                r[k] = m.as_double() if m is not None else float('nan')
            for k in ('l2_from_l1_rd', 'l2_from_l1_wr'):
                r[k] *= SECTOR
            for s in ('ld', 'st'):
                wf = r[f'smem_wf_{s}']
                r[f'smem_conflict_{s}'] = r[f'smem_bc_{s}'] / wf if wf and wf == wf else float('nan')
            rows.append(r)
    return rows


def load_trace(prefix, depth):
    """(phase, collapsed dst) -> logical inflow; plus collapsed edges for the graph."""
    inflow = defaultdict(lambda: [0.0, 0.0])
    edges = defaultdict(lambda: defaultdict(float))
    for path in glob.glob(f'{prefix}.dataflow_*.json'):
        d = json.load(open(path))
        ph = d['phase']
        for e in d['edges']:
            s, t = collapse(e['src'], depth), collapse(e['dst'], depth)
            if s == t:
                continue
            inflow[(ph, t)][0] += e['bytes']
            if s.startswith('param:'):
                inflow[(ph, t)][1] += e['bytes']
            edges[ph][(s, t)] += e['bytes']
    return inflow, edges


def nansum(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) if xs else float('nan')


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a = agg[(r['phase'], r['scope'])]
        for k in SUMMED:
            a[k].append(r[k])
        a['_n'].append(1)
        # hit rates: weight by the traffic they are a rate of
        a['_l2w'].append((r['l2_hit'], r['l2_from_l1_rd'] + r['l2_from_l1_wr']))
        a['_l1w'].append((r['l1_hit'], r['l1_glb_ld'] + r['l1_glb_st']))
    out = {}
    for key, a in agg.items():
        o = {k: nansum(a[k]) for k in SUMMED}
        o['kernels'] = len(a['_n'])
        for name, w in (('l2_hit', a['_l2w']), ('l1_hit', a['_l1w'])):
            ok = [(h, x) for h, x in w if h == h and x == x and x > 0]
            o[name] = sum(h * x for h, x in ok) / sum(x for _, x in ok) if ok else float('nan')
        for s in ('ld', 'st'):
            wf = o[f'smem_wf_{s}']
            o[f'smem_conflict_{s}'] = o[f'smem_bc_{s}'] / wf if wf and wf == wf else float('nan')
        out[key] = o
    return out


def mib(x):
    return x / 2**20


def write_dot(path, phase, edges, scopes, min_frac, pattern=None):
    tot = sum(edges.values()) or 1
    keep = {k: v for k, v in edges.items() if v / tot >= min_frac}
    if pattern:
        rx = re.compile(pattern)
        keep = {k: v for k, v in keep.items() if rx.search(k[0]) or rx.search(k[1])}
    nodes = {n for e in keep for n in e}
    with open(path, 'w') as f:
        f.write(f'digraph "{phase}" {{\n  rankdir=TB; node [shape=box, fontsize=10];\n')
        for n in sorted(nodes):
            s = scopes.get((phase, n))
            label = n
            if s:
                label += f'\\nDRAM rd {mib(s["dram_rd"]):.1f} MiB  {s["time_ns"] / 1e3:.0f} us'
                label += f'\\nL2 hit {s["l2_hit"]:.0f}%  smem {mib(s["smem_ld"] + s["smem_st"]):.1f} MiB'
            style = ', style=dashed' if n.startswith(('param:', 'state:')) else ''
            f.write(f'  "{n}" [label="{label}"{style}];\n')
        for (s, t), b in sorted(keep.items(), key=lambda kv: -kv[1]):
            f.write(f'  "{s}" -> "{t}" [label="{mib(b):.1f}"];\n')
        f.write('}\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('prefix')
    ap.add_argument('--depth', type=int, default=1, help='scope granularity, see collapse()')
    ap.add_argument('--dot-min-frac', type=float, default=0.002,
                    help='drop graph edges below this fraction of the phase total')
    ap.add_argument('--dot-filter', default=None,
                    help=r"regex; keep graph edges touching a matching scope, e.g. '^L(3|4)\b'")
    ap.add_argument('--top', type=int, default=25)
    a = ap.parse_args()

    reports = {c: f'{a.prefix}.{c}.ncu-rep' for c in ('all', 'none')
               if os.path.isfile(f'{a.prefix}.{c}.ncu-rep')}
    inflow, edges = load_trace(a.prefix, a.depth)
    if not reports and not edges:
        raise SystemExit(f'nothing found for prefix {a.prefix}')

    scopes, kernels = {}, []
    if reports:
        ncu_report = import_ncu_report()
        per = {}
        for cache, path in reports.items():
            rows = load_ncu(path, ncu_report, a.depth)
            for r in rows:
                r['cache'] = cache
            kernels += rows
            per[cache] = aggregate(rows)
            print(f'{path}: {len(rows)} kernels')
        scopes = per.get('all') or per['none']
        if 'all' in per and 'none' in per:
            for k, o in scopes.items():
                w = per['none'].get(k)
                o['dram_rd_warm'] = w['dram_rd'] if w else float('nan')
                o['warm_saved'] = o['dram_rd'] - o['dram_rd_warm']

    for (ph, sc), (lin, pin) in inflow.items():
        if (ph, sc) in scopes:
            o = scopes[(ph, sc)]
            o['logical_in'], o['param_in'] = lin, pin
            o['dram_per_logical'] = o['dram_rd'] / lin if lin else float('nan')

    if kernels:
        cols = ['cache', 'phase', 'scope', 'key', 'kernel'] + list(M) + ['smem_conflict_ld', 'smem_conflict_st']
        with open(f'{a.prefix}.kernels.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, cols, extrasaction='ignore')
            w.writeheader()
            w.writerows(kernels)
        print(f'wrote {a.prefix}.kernels.csv')

    if scopes:
        extra = ['logical_in', 'param_in', 'dram_per_logical', 'dram_rd_warm', 'warm_saved']
        cols = ['phase', 'scope', 'kernels'] + SUMMED + ['l2_hit', 'l1_hit',
                'smem_conflict_ld', 'smem_conflict_st'] + extra
        with open(f'{a.prefix}.scopes.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, cols, extrasaction='ignore')
            w.writeheader()
            for (ph, sc), o in sorted(scopes.items()):
                w.writerow({'phase': ph, 'scope': sc, **o})
        print(f'wrote {a.prefix}.scopes.csv')

        for ph in sorted({k[0] for k in scopes}):
            sel = sorted(((sc, o) for (p, sc), o in scopes.items() if p == ph),
                         key=lambda x: -x[1]['time_ns'])
            T = sum(o['time_ns'] for _, o in sel) or 1
            print(f'\n[{ph}] top scopes by kernel time (MiB; hit %; conflict = extra wavefronts)')
            print(f'{"scope":<44}{"time%":>6}{"DRAM rd":>9}{"wr":>7}{"L2->L1":>8}{"L2hit":>6}'
                  f'{"smem ld":>8}{"st":>7}{"conf":>6}{"logic_in":>9}{"warm_sv":>8}')
            for sc, o in sel[:a.top]:
                g = lambda k: o.get(k, float('nan'))
                print(f'{sc[:43]:<44}{100 * o["time_ns"] / T:>6.1f}{mib(g("dram_rd")):>9.1f}'
                      f'{mib(g("dram_wr")):>7.1f}{mib(g("l2_to_l1")):>8.1f}{g("l2_hit"):>6.0f}'
                      f'{mib(g("smem_ld")):>8.1f}{mib(g("smem_st")):>7.1f}{g("smem_conflict_ld"):>6.2f}'
                      f'{mib(g("logical_in")):>9.1f}{mib(g("warm_saved")):>8.1f}')

    for ph, e in edges.items():
        path = f'{a.prefix}.flow_{ph}.dot'
        write_dot(path, ph, e, scopes, a.dot_min_frac, a.dot_filter)
        print(f'wrote {path}   (render: dot -Tsvg {path} -o {path[:-4]}.svg)')


if __name__ == '__main__':
    main()
