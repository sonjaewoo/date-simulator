#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
config="${1:-hsim/configs/system.json}"
output_prefix="${2:-results/default}"
memory_config="${3:-hsim/configs/memory.json}"

case "$config" in
  /*) ;;
  *) config="$repo_root/$config" ;;
esac
case "$output_prefix" in
  /*) ;;
  *) output_prefix="$repo_root/$output_prefix" ;;
esac
case "$memory_config" in
  /*) ;;
  *) memory_config="$repo_root/$memory_config" ;;
esac

mkdir -p "$(dirname -- "$output_prefix")"

HSIM_SYSTEM_CONFIG="$config" \
HSIM_MEMORY_CONFIG="$memory_config" \
HSIM_CYCLE_LOG_FILE="$output_prefix.log" \
HSIM_PERFORMANCE_STATS_FILE="$output_prefix.breakdown.json" \
  "$repo_root/build/hsim/hsim" > "$output_prefix.stdout.log" 2>&1

tail -n 25 "$output_prefix.stdout.log"
