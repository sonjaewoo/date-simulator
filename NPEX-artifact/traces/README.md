# Control traces

Each subdirectory contains the `ssd_control.jsonl` file consumed by hsim.
GPU event traces and prompt/sample metadata are omitted because the simulator
does not read them.

Select a trace through `speculative.control_trace_file` in a system
configuration.  Relative paths are resolved from the repository root.
