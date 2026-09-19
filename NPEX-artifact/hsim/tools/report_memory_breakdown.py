#!/usr/bin/env python3
"""Summarize HSIM performance-breakdown JSON files as a Markdown table.

Example:
  python3 tools/report_memory_breakdown.py \
    PIM3_DRAM1=build/pim3_dram1.json \
    PIM2_DRAM2=build/pim2_dram2.json \
    PIM1_DRAM3=build/pim1_dram3.json
"""

import json
import sys
from pathlib import Path


def get(report, *path):
    value = report
    for key in path:
        value = value[key]
    return value


def pct(value):
    return f"{value:.1f}%"


def ns_to_us(value):
    return f"{value / 1_000:.1f}"


def diagnose(report):
    dram_busy = get(report, "dram", "memory", "busy_utilization_pct")
    pim_busy = get(report, "pim", "compute", "busy_utilization_pct")
    pim_hidden = get(report, "npu_pim_overlap", "pim_hidden_by_npu_pct")
    r_pim = report.get("workload_balance", {}).get("r_pim")
    eei = report.get("parallel_execution_imbalance", {}).get("aggregate", {}).get("eei")

    if r_pim is not None:
        if r_pim > 0.55:
            return "PIM/draft-heavy workload"
        if r_pim < 0.45:
            return "NPU/target-heavy workload"
        return "balanced busy-time workload"

    if eei is not None:
        if eei > 0.10:
            return "PIM-side exposed critical path"
        if eei < -0.10:
            return "NPU-side exposed critical path"
        return "balanced exposed execution"

    # Queue wait alone is not a bottleneck verdict: a long PIM batch may be
    # queued early yet still finish while target NPU work is running.  The
    # hidden-work ratio identifies whether that queue becomes tail latency.
    if pim_busy >= 85 and pim_hidden < 70:
        return "PIM tail / queueing bottleneck"
    if dram_busy >= 75 and pim_hidden >= 70:
        return "DRAM/NPU-side pressure; near balance"
    if abs(dram_busy - pim_busy) <= 15:
        return "balanced pressure"
    return "mixed / inspect queue wait"


def load_argument(value):
    label, separator, filename = value.rpartition("=")
    if not separator:
        filename = label
        label = Path(filename).stem
    with open(filename, encoding="utf-8") as handle:
        return label, json.load(handle)


def main(argv):
    if not argv:
        raise SystemExit("Pass one or more LABEL=breakdown.json files.")

    reports = [load_argument(argument) for argument in argv]
    print("| Split | DRAM ch. | PIM ch. | DRAM busy | DRAM BW (GB/s) | "
          "PIM busy | PIM queue p95 (us) | R_PIM | NPU covered | PIM hidden | EEI | Diagnosis |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for label, report in reports:
        print(
            f"| {label} | {get(report, 'dram', 'channels')} | "
            f"{get(report, 'pim', 'channels')} | "
            f"{pct(get(report, 'dram', 'memory', 'busy_utilization_pct'))} | "
            f"{get(report, 'dram', 'memory', 'achieved_bandwidth_GBps'):.2f} | "
            f"{pct(get(report, 'pim', 'compute', 'busy_utilization_pct'))} | "
            f"{ns_to_us(get(report, 'pim', 'compute', 'p95_queue_wait_ns'))} | "
            f"{report.get('workload_balance', {}).get('r_pim', float('nan')):.3f} | "
            f"{pct(get(report, 'npu_pim_overlap', 'npu_covered_by_pim_pct'))} | "
            f"{pct(get(report, 'npu_pim_overlap', 'pim_hidden_by_npu_pct'))} | "
            f"{report.get('parallel_execution_imbalance', {}).get('aggregate', {}).get('eei', float('nan')):.3f} | "
            f"{diagnose(report)} |"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
