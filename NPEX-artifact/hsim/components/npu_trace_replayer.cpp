#include "npu_trace_replayer.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>

sc_event npu_trace_done_ev;

namespace {

std::string episode_id_from_layer(const std::string& layer)
{
    const size_t sample = layer.find("sample");
    if (sample == std::string::npos) return {};
    const size_t iter = layer.find(".iter", sample);
    if (iter == std::string::npos) return {};
    size_t end = iter + 5;
    while (end < layer.size() && std::isdigit(static_cast<unsigned char>(layer[end]))) ++end;
    return layer.substr(sample, end - sample);
}

} // namespace

NPUTraceReplayer::NPUTraceReplayer(sc_module_name name, Logger* logger,
                                   PerformanceMetrics* metrics)
    : sc_module(name), master("master"), slave("slave"), clock("clock"),
      peq_fw(this, &NPUTraceReplayer::peq_fw_cb),
      peq_bw(this, &NPUTraceReplayer::peq_bw_cb),
      profile(cfgs.get_synthetic_npu_info()),
      layer_profile(cfgs.get_target_layer_profile()),
      dram_req_bytes(cfgs.get_dram_req_size()), logger(logger), metrics(metrics)
{
    slave.register_nb_transport_fw(this, &NPUTraceReplayer::nb_transport_fw);
    master.register_nb_transport_bw(this, &NPUTraceReplayer::nb_transport_bw);
    SC_THREAD(replay_trace);
}

tlm_sync_enum NPUTraceReplayer::nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    peq_fw.notify(trans, phase, SC_ZERO_TIME);
    return TLM_UPDATED;
}

tlm_sync_enum NPUTraceReplayer::nb_transport_bw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    peq_bw.notify(trans, phase, SC_ZERO_TIME);
    return TLM_UPDATED;
}

void NPUTraceReplayer::peq_fw_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    if (phase == BEGIN_REQ && trans.get_command() == TLM_COMPUTE_COMMAND) {
        active_episode_id = episode_id_from_layer(trans.get_layer());
        active_context_tokens = std::max(1u, trans.get_data_length());
        active_query_tokens = std::max(1u, trans.get_bst_size());
        active_kv_write_tokens = std::max(1u, static_cast<uint32_t>(trans.get_address()));
        start_event.notify(SC_ZERO_TIME);
    }
}

uint64_t NPUTraceReplayer::operation_compute_ns(const ModelOperationInfo& op) const
{
    if (!profile.analytical_compute_enabled) return 0;
    const long double query_tokens = active_query_tokens;
    const long double context_tokens = active_context_tokens;
    const long double macs = query_tokens * op.macs_per_token
        + query_tokens * context_tokens * op.macs_per_kv_token;
    if (macs == 0.0L) return 0;
    const long double cycles = std::ceil(macs / profile.macs_per_cycle);
    // GHz is cycles/ns: 0.5 GHz therefore gives 2 ns per cycle.
    return static_cast<uint64_t>(std::ceil(cycles / profile.compute_frequency_GHz));
}

void NPUTraceReplayer::peq_bw_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    if (phase == BEGIN_RESP) {
        complete_memory_response(trans);
    }
}

void NPUTraceReplayer::complete_memory_response(tlm_generic_payload& trans)
{
    tlm_phase phase = END_RESP;
    const tlm_sync_enum reply = master->nb_transport_fw(trans, phase, t);
    assert(reply == TLM_COMPLETED);
    trans.release();
    assert(pending_responses > 0);
    --pending_responses;
    response_event.notify(SC_ZERO_TIME);
}

void NPUTraceReplayer::issue_memory(tlm_command command, uint32_t bytes, const std::string& layer)
{
    const uint32_t request_bytes = std::max<uint32_t>(1, dram_req_bytes);
    uint32_t remaining = bytes;
    uint32_t burst_id = 0;
    const uint32_t burst_size = (bytes + request_bytes - 1) / request_bytes;
    if (burst_size == 0) return;
    pending_responses = 0;

    while (remaining > 0) {
        const uint32_t payload_bytes = std::min(remaining, request_bytes);
        tlm_generic_payload* trans = m_mm.allocate();
        trans->acquire();
        trans->set_command(command);
        trans->set_src_id(NPU);
        trans->set_dst_id(cfgs.dram_enabled() ? DRAM : PIM);
        trans->set_address(next_address);
        trans->set_data_length(payload_bytes);
        trans->set_bst_id(burst_id++);
        trans->set_bst_size(burst_size);
        trans->set_layer(layer);

        // Synthetic addresses avoid every request aliasing a single DRAM row.
        next_address += request_bytes;
        tlm_phase phase = BEGIN_REQ;
        const tlm_sync_enum reply = master->nb_transport_fw(*trans, phase, t);
        assert(reply == TLM_UPDATED);
        ++pending_responses;
        remaining -= payload_bytes;
        // A 7B FP16 MLP projection spans millions of 64-B DRAM requests.
        // Keep the exact byte/request count, but bound payload ownership.
        if (pending_responses >= profile.max_inflight_requests) {
            wait(response_event);
        }
    }

    // Drain the final DMA window before the operation's compute phase.
    while (pending_responses > 0) {
        wait(response_event);
    }
}

void NPUTraceReplayer::issue_fast_memory(tlm_command command, uint64_t bytes, const std::string& layer)
{
    if (bytes == 0) return;

    // Send one tensor-sized transaction through the same Arbiter path as the
    // detailed model.  Its data_length/burst_size pair carries a 64-bit byte
    // count for the wrapper's calibrated DRAMsim3 macro model; the payload is
    // never decomposed into individual cache-line TLM transactions here.
    tlm_generic_payload* trans = m_mm.allocate();
    trans->acquire();
    trans->set_command(command);
    trans->set_src_id(NPU);
    trans->set_dst_id(cfgs.dram_enabled() ? DRAM : PIM);
    trans->set_address(next_address);
    trans->set_data_length(static_cast<uint32_t>(bytes & 0xffffffffULL));
    trans->set_bst_id(0);
    trans->set_bst_size(static_cast<uint32_t>(bytes >> 32U));
    trans->set_layer(layer);
    trans->set_pim_cmd("fast_memory");
    next_address += bytes;
    pending_responses = 1;
    tlm_phase phase = BEGIN_REQ;
    const tlm_sync_enum reply = master->nb_transport_fw(*trans, phase, t);
    assert(reply == TLM_UPDATED);
    while (pending_responses > 0) wait(response_event);
}

void NPUTraceReplayer::replay_trace()
{
    while (true) {
        wait(start_event);
        active_core = 1;
        const uint32_t current_run = run_id++;
        const uint32_t layer_count = layer_profile.layer_count;
        uint32_t total_compute_share = 0;
        for (const auto& op : layer_profile.operations) total_compute_share += op.compute_share;
        std::cout << "[SyntheticNPU] target trace " << current_run
                  << ": " << layer_profile.name << ", " << layer_count
                  << " decoder layers, " << layer_profile.operations.size()
                  << " ops/layer"
                  << " query_tokens=" << active_query_tokens
                  << " kv_tokens=" << active_context_tokens
                  << " cycle=" << static_cast<int>(sc_time_stamp().to_double() / 1000) << "\n";

        if (profile.fast_layer_mode) {
            // "layer" granularity keeps the analytical bandwidth model but
            // walks the decoder layers one at a time, so the cycle log gets a
            // record per layer instead of a single collapsed macro record.
            const bool collapse_layers = profile.fast_layer_granularity == "macro";
            const uint32_t simulated_layer_count = collapse_layers ? 1 : layer_profile.layer_count;
            const uint32_t layer_multiplier = collapse_layers ? layer_profile.layer_count : 1;

            for (uint32_t layer_id = 0; layer_id < simulated_layer_count; ++layer_id) {
                const std::string layer = "target_npu.run" + std::to_string(current_run)
                    + (collapse_layers ? ".macro" : ".layer" + std::to_string(layer_id));
                const sc_time npu_start_time = sc_time_stamp();
                logger->update_npu_start(layer, static_cast<int64_t>(sc_time_stamp().to_double() / 1000));
                uint64_t assigned_compute = 0;
                for (size_t op_id = 0; op_id < layer_profile.operations.size(); ++op_id) {
                    const auto& op = layer_profile.operations[op_id];
                    const std::string op_layer = layer + ".op." + op.name;
                    if (op.weight_read_bytes > 0) {
                        const uint64_t bytes = op.weight_read_bytes * layer_multiplier;
                        std::cout << "[SyntheticNPU][FAST_READ] " << op_layer
                                  << " weight=" << bytes << "B\n";
                        issue_fast_memory(TLM_READ_COMMAND, bytes, op_layer + ".weight");
                    }
                    if (op.name == "qkt" && layer_profile.kv_cache_bytes_per_token > 0) {
                        const uint64_t bytes = layer_profile.kv_cache_bytes_per_token
                            * active_context_tokens * layer_multiplier;
                        issue_fast_memory(TLM_READ_COMMAND, bytes, op_layer + ".kv_cache.read");
                    }
                    const uint64_t per_layer_compute = profile.analytical_compute_enabled
                        ? operation_compute_ns(op)
                        : ((op_id + 1 == layer_profile.operations.size())
                            ? profile.compute_ns_per_layer - assigned_compute
                            : (profile.compute_ns_per_layer * op.compute_share) / total_compute_share);
                    assigned_compute += per_layer_compute;
                    const uint64_t compute_ns = static_cast<uint64_t>(per_layer_compute)
                        * layer_multiplier;
                    const sc_time compute_start = sc_time_stamp();
                    const sc_time compute_end = compute_start + sc_time(compute_ns, SC_NS);
                    metrics->record_npu_compute_interval(compute_start, compute_end);
                    wait(compute_ns, SC_NS);
                    if (op.name == "v_proj" && layer_profile.kv_cache_bytes_per_token > 0) {
                        const uint64_t bytes = layer_profile.kv_cache_bytes_per_token
                            * active_kv_write_tokens * layer_multiplier;
                        issue_fast_memory(TLM_WRITE_COMMAND, bytes, op_layer + ".kv_cache.write");
                    }
                }
                // output_write_bytes is specified per decoder token.  A
                // PEARL post-verify forward materializes one output for each
                // uncached speculative token, while weights remain reusable
                // across that batch.
                const uint64_t output_bytes = (layer_profile.output_write_bytes != 0
                    ? layer_profile.output_write_bytes : profile.output_write_bytes_per_layer)
                    * active_query_tokens * layer_multiplier;
                issue_fast_memory(TLM_WRITE_COMMAND, output_bytes, layer + ".output");
                const int64_t end_cycle = static_cast<int64_t>(sc_time_stamp().to_double() / 1000);
                logger->update_npu_end(layer, end_cycle,
                                       assigned_compute * layer_multiplier);
                metrics->record_npu_interval(npu_start_time, sc_time_stamp());
                std::cout << "[SyntheticNPU][FAST] completed " << layer
                          << " (" << layer_multiplier << " decoder layers) cycle=" << end_cycle << "\n";
            }
            active_core = 0;
            metrics->complete_parallel_npu_episode(active_episode_id, sc_time_stamp());
            npu_trace_done_ev.notify(SC_ZERO_TIME);
            continue;
        }

        for (uint32_t layer_id = 0; layer_id < layer_count; ++layer_id) {
            const std::string layer = "target_npu.run" + std::to_string(current_run)
                                    + ".layer" + std::to_string(layer_id);
            const sc_time npu_start_time = sc_time_stamp();
            logger->update_npu_start(layer, static_cast<int64_t>(sc_time_stamp().to_double() / 1000));
            uint64_t assigned_compute = 0;
            for (size_t op_id = 0; op_id < layer_profile.operations.size(); ++op_id) {
                const auto& op = layer_profile.operations[op_id];
                const std::string op_layer = layer + ".op." + op.name;
                if (op.weight_read_bytes > 0) {
                    std::cout << "[SyntheticNPU][READ] " << op_layer
                              << " weight=" << op.weight_read_bytes << "B"
                              << " cycle=" << static_cast<int>(sc_time_stamp().to_double() / 1000) << "\n";
                    issue_memory(TLM_READ_COMMAND, static_cast<uint32_t>(op.weight_read_bytes),
                                 op_layer + ".weight");
                }
                if (op.name == "qkt" && layer_profile.kv_cache_bytes_per_token > 0) {
                    const uint64_t bytes = layer_profile.kv_cache_bytes_per_token * active_context_tokens;
                    issue_memory(TLM_READ_COMMAND, static_cast<uint32_t>(bytes), op_layer + ".kv_cache.read");
                }
                const uint64_t compute_ns = profile.analytical_compute_enabled
                    ? operation_compute_ns(op)
                    : ((op_id + 1 == layer_profile.operations.size())
                        ? profile.compute_ns_per_layer - assigned_compute
                        : (profile.compute_ns_per_layer * op.compute_share) / total_compute_share);
                assigned_compute += compute_ns;
                if (compute_ns > 0) {
                    const sc_time compute_start = sc_time_stamp();
                    metrics->record_npu_compute_interval(
                        compute_start, compute_start + sc_time(compute_ns, SC_NS));
                    wait(compute_ns, SC_NS);
                }
                if (op.name == "v_proj" && layer_profile.kv_cache_bytes_per_token > 0) {
                    const uint64_t bytes = layer_profile.kv_cache_bytes_per_token * active_kv_write_tokens;
                    issue_memory(TLM_WRITE_COMMAND, static_cast<uint32_t>(bytes), op_layer + ".kv_cache.write");
                }
            }
            const uint64_t output_bytes = (layer_profile.output_write_bytes != 0
                ? layer_profile.output_write_bytes : profile.output_write_bytes_per_layer)
                * active_query_tokens;
            if (output_bytes > 0) {
                issue_memory(TLM_WRITE_COMMAND, static_cast<uint32_t>(output_bytes), layer + ".output");
            }
            const int64_t end_cycle = static_cast<int64_t>(sc_time_stamp().to_double() / 1000);
            logger->update_npu_end(layer, end_cycle, assigned_compute);
            metrics->record_npu_interval(npu_start_time, sc_time_stamp());
            std::cout << "[SyntheticNPU] completed " << layer
                      << " cycle=" << end_cycle << "\n";
        }
        active_core = 0;
        metrics->complete_parallel_npu_episode(active_episode_id, sc_time_stamp());
        npu_trace_done_ev.notify(SC_ZERO_TIME);
    }
}
