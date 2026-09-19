#pragma once

#include <components/trace_generator.h>

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_speculative_s1(TraceGenerator& gen);
