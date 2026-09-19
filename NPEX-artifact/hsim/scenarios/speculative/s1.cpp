#include "s1.h"

std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_speculative_s1(TraceGenerator& gen)
{
    gen.add_speculative_trace("pearl_s1");
    gen.add_trace(TraceType::TERMINATE, "", HOST);
    return gen.trace_queue;
}
