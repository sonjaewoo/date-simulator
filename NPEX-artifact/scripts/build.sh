#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

cmake -S "$repo_root" -B "$repo_root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$repo_root/build" --parallel "${NPEX_BUILD_JOBS:-4}"
