#!/usr/bin/env python3
import json
import sys
from pathlib import Path


EXPECTED_SIMULATION_NS = 67_628_880_001


def main() -> int:
    result = Path(sys.argv[1] if len(sys.argv) > 1 else "results/default.breakdown.json")
    with result.open(encoding="utf-8") as handle:
        data = json.load(handle)
    actual = data.get("simulation_ns")
    if actual != EXPECTED_SIMULATION_NS:
        print(f"FAIL: simulation_ns={actual}, expected={EXPECTED_SIMULATION_NS}")
        return 1
    print(f"PASS: simulation_ns={actual}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
