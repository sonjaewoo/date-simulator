# Memory-pressure breakdown

Each `hsim` run now writes `<cycle_log_file>.breakdown.json` by default.  Set
`performance_stats_file` in the system JSON when the report should use an
explicit path.  `HSIM_CYCLE_LOG_FILE` and `HSIM_PERFORMANCE_STATS_FILE` offer
the same per-run overrides without editing a shared experiment configuration.

The report separates three effects:

- `workload_balance`: the primary cross-workload comparison metric.  It is
  `r_pim = pim_draft_busy_ns / (pim_draft_busy_ns + npu_target_busy_ns)`.
  `pim_draft_busy_ns` contains only draft PIM command service and
  `npu_target_busy_ns` only target NPU compute intervals; queueing,
  memory-response waits, and inter-device dependency stalls are excluded.
  Compare this value only across workloads executed at one fixed reference
  partition (normally PIM:DRAM=2:2).  `r_pim > 0.5` means relatively more
  draft/PIM demand, while `r_pim < 0.5` means target/NPU demand dominates.

- `dram.memory`: target-side transferred bytes, achieved bandwidth, wrapper
  queue wait, and device/channel busy time.
- `pim.compute`: draft PIM-engine occupancy and its request queue wait.  In
  unified memory, `pim_side_stall` additionally isolates direct contention
  between target memory access and draft compute.
- `npu_pim_overlap`: intersection of target NPU-layer execution and PIM
  compute.  `npu_covered_by_pim_pct` says how much target work ran alongside
  PIM; `pim_hidden_by_npu_pct` says how much PIM work was hidden by target
  execution.  Use both: a high first value with a low second value means PIM
  continues after NPU and is still the tail bottleneck.
- `parallel_execution_imbalance`: episode-level critical-path decomposition.
  Each speculative stage has a common launch point; from there, `both_ns` is
  time before either branch completes, while `pim_only_ns` and `npu_only_ns`
  are the exposed tails.  The aggregate EEI is
  `(pim_only_ns - npu_only_ns) / (pim_only_ns + npu_only_ns + both_ns)`.
  Positive EEI identifies an exposed PIM-side critical path, negative EEI an
  exposed NPU-side critical path, and values near zero a balanced stage.  This
  branch-window metric deliberately includes PIM queueing and target memory
  waits, because both delay the speculative-stage barrier.  It is a separate
  performance-tail diagnosis, not the busy-time workload-balance metric.

Fast-layer tensor macros use an aggregate calibrated memory model.  Their
per-channel entries are therefore a balanced logical-channel estimate, marked
by `channel_accounting`; detailed DRAMsim3 runs report the physical address
mapped channel for each request.

Compare splits with:

```bash
python3 tools/report_memory_breakdown.py \
  'PIM:DRAM=3:1'=build/pim3_dram1.breakdown.json \
  'PIM:DRAM=2:2'=build/pim2_dram2.breakdown.json \
  'PIM:DRAM=1:3'=build/pim1_dram3.breakdown.json
```

The existing `configs/experiments/memory_partitioned_3_1.json` convention is
**DRAM:PIM**, not PIM:DRAM: it contains DRAM=3 and PIM=1.  Either label results
with that convention or swap the two config files when presenting a
PIM:DRAM sweep.
