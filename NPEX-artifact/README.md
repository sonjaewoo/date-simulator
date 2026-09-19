# NPEX-extended

This repository contains the trace-driven simulator used to evaluate
speculative decoding with an NPU target model and a PIM draft model.  It is a
standalone artifact: the control traces, model profiles, DRAMsim3 source, and
the required SystemC/JsonCpp runtime files are included.

## Requirements

- Linux x86-64
- CMake 3.14 or newer
- A C++20 compiler (tested with GCC 11)
- Python 3 (only for the result verification helper)

## Build

```bash
./scripts/build.sh
```

This builds DRAMsim3 and hsim from source under `build/`.

## Run the default experiment

```bash
./scripts/run.sh
```

The default configuration is `hsim/configs/system.json`.  Results are written
to `results/default.log` and `results/default.breakdown.json`.  Verbose
simulator output is captured in `results/default.stdout.log`; the final summary
is printed in the terminal.

To run another configuration and choose an output prefix:

```bash
./scripts/run.sh \
  hsim/configs/experiments/system_energy_fast_partitioned.json \
  results/fast_partitioned \
  hsim/configs/experiments/memory_partitioned_2_2.json
```

Paths inside configuration files are resolved relative to the repository
root.  To use another included trace, set `speculative.control_trace_file` to
`traces/<trace-name>/ssd_control.jsonl`.

## Verify the reference result

```bash
./scripts/verify_default.py results/default.breakdown.json
```


## Repository layout

- `hsim/`: trace replay, NPU/PIM timing models, and experiment configurations
- `traces/`: control JSONL traces consumed by hsim
- `DRAMsim3/`: DRAM timing backend with the NPEX integration
- `libs/`: bundled SystemC and JsonCpp headers/libraries
