# Validation

The artifact was built from a clean build directory and compared with the
working NPEX tree on 2026-09-19.

| Configuration | Working tree (ns) | Artifact (ns) | Difference |
|---|---:|---:|---:|
| Default unified | 67,628,880,001 | 67,628,880,001 | 0 |
| Fast unified | 15,710,490,001 | 15,710,490,001 | 0 |
| Fast partitioned (2 DRAM + 2 PIM channels) | 15,128,895,332 | 15,128,895,696 | 364 ns (0.0000024%) |

The working tree's checked-in `libdramsim3.so` was linked from a mixture of
older object files, while this artifact rebuilds DRAMsim3 entirely from the
included source.  This accounts for the 364 ns partitioned-memory difference.
The source-built artifact value is the public reproducibility reference.

The default result can be checked with:

```bash
./scripts/run.sh
./scripts/verify_default.py results/default.breakdown.json
```
