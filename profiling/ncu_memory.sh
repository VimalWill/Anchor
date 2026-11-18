#!/usr/bin/env bash
# Nsight Compute memory-hierarchy capture of one inference request on GH200.
#
#   bash profiling/ncu_memory.sh OUT_PREFIX [profile_inference.py args...]
#   bash profiling/ncu_memory.sh prof/run --cfg CFG --ckpt CKPT --seq-len 2048 --layers 0,-1
#
# Writes OUT_PREFIX.<cache>.ncu-rep (open it in ncu-ui, Memory Workload Analysis
# chart = the L1/shared <-> L2 <-> DRAM flow diagram per kernel) and
# OUT_PREFIX.ncu-meta.json.
#
# Environment knobs:
#   CACHE=all   (default) flush L1/L2 before every kernel. Each kernel's DRAM bytes
#               are what it needs with nothing resident: per-kernel cost, upper bound.
#   CACHE=none  keep caches. DRAM bytes now exclude what earlier kernels left in L2.
#               Run both: analyze_ncu.py reports cold - warm DRAM bytes per scope,
#               the data a function received from its producers through L2.
#               Caveat: with REPLAY=kernel, replay passes 2+ of a kernel see the L2
#               state left by its own pass 1, so warm hit rates are optimistic.
#               REPLAY=application avoids that by rerunning the whole process
#               per pass (model load each time: slow, but faithful).
#   REPLAY=kernel|application   (default kernel)
#   PYTHON_BIN=python3
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${1:?usage: ncu_memory.sh OUT_PREFIX [profile_inference.py args...]}
shift
CACHE=${CACHE:-all}
REPLAY=${REPLAY:-kernel}
PYTHON_BIN=${PYTHON_BIN:-python3}
export PYTHONPATH=".:${PYTHONPATH:-}"
mkdir -p "$(dirname "$OUT")"

command -v ncu >/dev/null || { echo 'ncu not on PATH (module load cuda / nsight-compute)'; exit 1; }
ncu --version | tail -1

# Per-kernel bytes at each level of the hierarchy. Sectors are 32 B.
WANT=(
  gpu__time_duration.sum
  dram__bytes_read.sum dram__bytes_write.sum
  gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed
  lts__t_sectors_srcunit_tex_op_read.sum lts__t_sectors_srcunit_tex_op_write.sum
  lts__t_sector_hit_rate.pct
  l1tex__m_xbar2l1tex_read_bytes.sum l1tex__m_l1tex2xbar_write_bytes.sum
  l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum
  l1tex__t_sector_hit_rate.pct
  sm__sass_data_bytes_mem_shared_op_ld.sum sm__sass_data_bytes_mem_shared_op_st.sum
  l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum l1tex__data_pipe_lsu_wavefronts_mem_shared_op_st.sum
  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum
)
# Drop any metric this ncu does not know for GH100, instead of failing after model load.
AVAIL=$(ncu --query-metrics --chip gh100 2>/dev/null | awk '{print $1}')
METRICS=()
if [ -z "$AVAIL" ]; then
  echo 'warning: ncu --query-metrics returned nothing; requesting every metric unchecked'
  METRICS=("${WANT[@]}")
else
  for m in "${WANT[@]}"; do
    if grep -qxF "${m%%.*}" <<<"$AVAIL"; then METRICS+=("$m"); else echo "skip unknown metric: $m"; fi
  done
fi
METRIC_ARG=$(IFS=,; echo "${METRICS[*]}")

set -x
ncu \
  --target-processes all \
  --profile-from-start off \
  --nvtx \
  --kernel-name-base demangled \
  --section SpeedOfLight \
  --section MemoryWorkloadAnalysis \
  --section MemoryWorkloadAnalysis_Chart \
  --section MemoryWorkloadAnalysis_Tables \
  --section LaunchStats \
  --section Occupancy \
  --metrics "$METRIC_ARG" \
  --cache-control "$CACHE" \
  --replay-mode "$REPLAY" \
  --clock-control base \
  -f -o "${OUT}.${CACHE}" \
  "$PYTHON_BIN" profiling/profile_inference.py --mode ncu --out "$OUT" "$@" \
|| { set +x; echo; echo 'If the error is ERR_NVGPUCTRPERM, GPU performance counters are'
     echo 'restricted to admins on this node; ask the cluster admins for profiling access.'; exit 1; }
