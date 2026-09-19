#include "host.h"

#include <cmath>

bool is_pim_busy = false;
sc_event pim_compute_done_ev;
sc_event pim_trace_done_ev;

Host::Host(sc_module_name name, Logger* logger, PerformanceMetrics* metrics)
: master("master"), clock("clock"), peq(this, &Host::peq_cb), logger(logger), metrics(metrics)
{
    SC_THREAD(process_trace);
    master.register_nb_transport_bw(this, &Host::nb_transport_bw);

    SC_METHOD(clock_negedge);
    sensitive << clock.neg();
    dont_initialize();

    dram_req_bytes = cfgs.get_dram_req_size();
    pim_wait_map = cfgs.get_pim_wait_map();
    pim_sync_map = cfgs.get_pim_sync_map();
    host_profile = cfgs.get_host_profile();
    pim_profile = cfgs.get_pim_profile();
    tile_cnt_by_layer = cfgs.get_layer_tile_map();
    
    trace_q = trace_generator.generate_trace();
};

void Host::clock_negedge()
{
    /* No pending transactions */
    if (pending_q.empty()) {
        return;
    }
    
    /* Dequeue next transaction */
    tlm_generic_payload* trans = pending_q.front();
    pending_q.pop_front();

    /* Select target socket (0 = NPU, 1 = Arbiter) */
    int id = (trans->get_dst_id() == NPU) ? 0 : 1;

    switch (trans->get_dst_id()) {
        case NPU:
            id = 0;
            break;
        case DRAM:
        case PIM:
            id = 1;
            break;
    }

    if (id == 1) { print_log(trans); }

    tlm_phase phase = BEGIN_REQ;
    tlm_sync_enum reply = master[id]->nb_transport_fw(*trans, phase, t);
    assert(reply == TLM_UPDATED);

    active_cycle++;
}

void Host::process_trace()
{
    while (true) {
        if (!trace_q.empty()) {
            auto trace = trace_q.front();
            trace_q.pop_front();

            switch (trace->type) {
                case TraceType::WRITE:
                case TraceType::READ:
                    handle_memory_trace(std::dynamic_pointer_cast<TraceGenerator::MemoryTrace>(trace));
                    break;
                case TraceType::COMPUTE:
                    handle_compute_trace(std::dynamic_pointer_cast<TraceGenerator::ComputeTrace>(trace));
                    break;
                case TraceType::PIM:
                    handle_pim_trace(std::dynamic_pointer_cast<TraceGenerator::PimTrace>(trace));
                    break;
                case TraceType::SPECULATIVE:
                    handle_speculative_trace(std::dynamic_pointer_cast<TraceGenerator::SpeculativeTrace>(trace));
                    break;
                case TraceType::TERMINATE:
                    active_host = 0;
                    break;
            }
        }
        if (active_dram == 0) break;
        wait(1, SC_NS);
    }
    logger->print_and_save_log(cfgs.get_cycle_log_file()); /* Save into file */
    sc_stop();
}

void Host::handle_speculative_trace(const std::shared_ptr<TraceGenerator::SpeculativeTrace>& trace)
{
    const auto spec = cfgs.get_speculative_info();
    const auto& controls = trace_generator.get_speculative_control();
    const auto synthetic_pim = cfgs.get_synthetic_pim_info();
    const auto synthetic_npu = cfgs.get_synthetic_npu_info();
    const auto& target_layer_profile = cfgs.get_target_layer_profile();
    const auto& draft_layer_profile = cfgs.get_draft_layer_profile();
    const auto profile_it = pim_profile.find(spec.draft_profile_layer);
    const auto tile_it = tile_cnt_by_layer.find(spec.draft_profile_layer);
    if (!synthetic_pim.enabled && (profile_it == pim_profile.end() || tile_it == tile_cnt_by_layer.end())) {
        throw std::runtime_error("Missing PIM draft template: " + spec.draft_profile_layer);
    }
    std::cout << "[Speculative] scenario=" << trace->layer
              << ", control iterations=" << controls.size() << "\n";

    for (const auto& control : controls) {
        const std::string sample_prefix = "sample" + std::to_string(control.sample_index)
                                        + ".iter" + std::to_string(control.iteration);
        const bool ssd_pipeline = control.algorithm == "ssd" && control.draft_overlaps_target;
        struct DraftWorkItem {
            uint32_t context_tokens;
            uint32_t tree_step;
            uint32_t branch_depth;
            uint32_t branch_index;
        };
        std::vector<DraftWorkItem> draft_work_items;
        if (control.algorithm == "ssd") {
            // The SSD runtime processes a full fan-out batch at each tree
            // decode step.  This simulator deliberately serializes that
            // batch on the PIM command engine, but does not turn it into a
            // linear token chain: siblings retain the KV length of their
            // originating verification position (branch_depth), and each
            // following tree step extends only its own branch.
            for (uint32_t tree_step = 0; tree_step < control.draft_tree_steps; ++tree_step) {
                for (uint32_t branch_depth = 0;
                     branch_depth < control.draft_fan_out_list.size(); ++branch_depth) {
                    for (uint32_t branch_index = 0;
                         branch_index < control.draft_fan_out_list[branch_depth]; ++branch_index) {
                        draft_work_items.push_back({
                            control.draft_kv_cache_tokens + branch_depth + tree_step,
                            tree_step,
                            branch_depth,
                            branch_index,
                        });
                    }
                }
            }
        } else {
            // PEARL's draft is a normal autoregressive proposal sequence.
            for (uint32_t proposal = 0; proposal < control.draft_proposal_count; ++proposal) {
                draft_work_items.push_back({
                    control.draft_kv_cache_tokens + proposal,
                    0,
                    proposal,
                    0,
                });
            }
        }
        const std::string barrier_layer = (ssd_pipeline ? "ssd." : "speculative.")
                                        + sample_prefix + ".barrier";
        metrics->begin_parallel_episode(sample_prefix, sc_time_stamp());
        logger->update_start(barrier_layer, HOST);
        std::cout << (ssd_pipeline ? "[SSD][Pipeline][Launch]" : "[Speculative][Launch]")
                  << " sample=" << control.sample_index
                  << " iter=" << control.iteration
                  << " prefix=" << control.prefix_len_before
                  << " gamma=" << control.gamma
                  << " target_q=" << control.target_query_tokens
                  << " target_kv=" << control.target_kv_cache_tokens
                  << " draft_proposals=" << control.draft_proposal_count
                  << " draft_kv=" << control.draft_kv_cache_tokens
                  << " trace_v=" << control.trace_version
                  << (control.algorithm == "ssd" ? " tree_steps="
                                      + std::to_string(control.draft_tree_steps)
                                      + " serial_tree_nodes=" + std::to_string(draft_work_items.size())
                                   : "")
                  << (ssd_pipeline ? " draft_precompute_for="
                                      + std::to_string(control.draft_precompute_for_iteration)
                                   : "")
                  << " mode=" << control.mode_before
                  << " | target=" << target_layer_profile.name << " ("
                  << target_layer_profile.layer_count << " layers, "
                  << target_layer_profile.operations.size() << " ops/layer)"
                  << " | draft=" << draft_layer_profile.name << " ("
                  << draft_layer_profile.layer_count << " layers, "
                  << draft_layer_profile.operations.size() << " ops/layer)"
                  << " cycle=" << current_cycle() << "\n";
        // Start target first, then enqueue the draft PIM work.  For SSD this
        // is the asynchronous pipeline stage: Draft(r) precomputes the tree
        // used by Target(r+1) while Target(r) verifies.  PIM remains a
        // sequential command engine; only the independent NPU/PIM actors
        // overlap.  Host issues one TLM request per clock, so this ordering
        // prevents a long draft command stream from delaying target launch.
        tlm_generic_payload* npu_trans = m_mm.allocate();
        npu_trans->acquire();
        npu_trans->set_command(TLM_COMPUTE_COMMAND);
        npu_trans->set_src_id(HOST);
        npu_trans->set_dst_id(NPU);
        npu_trans->set_layer("target." + sample_prefix);
        // Carry the target verification batch and its KV context to the
        // synthetic NPU.  They are used only by the analytical MAC model.
        npu_trans->set_data_length(control.target_kv_cache_tokens);
        npu_trans->set_bst_size(control.target_query_tokens);
        npu_trans->set_address(control.target_kv_write_tokens);
        pending_q.push_back(npu_trans);

        // Both actors then share the existing Arbiter/DRAM backplane rather
        // than being serialized by the speculative controller.
        if (synthetic_pim.enabled) {
            if (draft_layer_profile.layer_count == 0) {
                throw std::runtime_error("Synthetic PIM model profile has no decoder layers");
            }

            // LPDDR5.json maps model-operation names to a PIM operation
            // profile.  In aggregation mode its whole tile-expanded stream
            // becomes one PIM transaction per model operation.
            const bool aggregate_micro = synthetic_pim.aggregate_micro_commands
                                      || synthetic_pim.fast_layer_mode;
            // "layer" granularity keeps the aggregated micro-command timing
            // model but issues one macro per decoder layer, so the cycle log
            // carries a record per layer instead of a single collapsed one.
            const bool collapse_layers = synthetic_pim.fast_layer_mode
                                      && synthetic_pim.fast_layer_granularity == "macro";
            const uint32_t simulated_layer_count = collapse_layers
                                      ? 1 : draft_layer_profile.layer_count;
            const uint32_t layer_multiplier = collapse_layers
                                      ? draft_layer_profile.layer_count : 1;
            const auto draft_attention_tiles = [&](const ModelOperationInfo& op, const DraftWorkItem& work) {
                if (op.name != "qkt" && op.name != "sv") return 1u;
                const uint32_t context = work.context_tokens;
                return std::max(1u, (context + draft_layer_profile.attention_context_tile_tokens - 1)
                    / draft_layer_profile.attention_context_tile_tokens);
            };
            uint64_t per_decoder_layer_transactions = 0;
            for (const auto& work : draft_work_items) {
              for (const auto& op : draft_layer_profile.operations) {
                const auto profile_it = pim_profile.find(op.name);
                const auto tile_it = tile_cnt_by_layer.find(op.name);
                if (profile_it == pim_profile.end() || tile_it == tile_cnt_by_layer.end()) {
                    std::cout << "[SyntheticPIM][Skip] " << op.name
                              << " has no LPDDR5 micro-operation profile\n";
                    continue;
                }
                if (aggregate_micro) {
                    ++per_decoder_layer_transactions;
                } else {
                    uint64_t commands_per_tile = 0;
                    for (const auto& command_name : profile_it->second.ordered_cmd_name) {
                        commands_per_tile += profile_it->second.cmd_map.at(command_name).count;
                    }
                    per_decoder_layer_transactions += tile_it->second * draft_attention_tiles(op, work) * commands_per_tile;
                }
              }
            }
            // per_decoder_layer_transactions already includes every draft
            // proposal.  Multiply only by the number of simulated decoder
            // layers; multiplying by it again prevents the final macro from
            // ever receiving last=true and deadlocks the Host barrier.
            const uint64_t total_transactions = per_decoder_layer_transactions
                                              * simulated_layer_count;
            if (total_transactions == 0) {
                throw std::runtime_error("No draft PIM operations match the active LPDDR5 profile");
            }
            uint64_t transaction_index = 0;
            for (const auto& work : draft_work_items) {
                for (uint32_t layer_id = 0; layer_id < simulated_layer_count; ++layer_id) {
                    for (const auto& op : draft_layer_profile.operations) {
                        const auto profile_it = pim_profile.find(op.name);
                        const auto tile_it = tile_cnt_by_layer.find(op.name);
                        if (profile_it == pim_profile.end() || tile_it == tile_cnt_by_layer.end()) continue;
                        const uint32_t tile_count = tile_it->second * draft_attention_tiles(op, work);
                        const auto& micro_profile = profile_it->second;
                        const std::string layer = "draft_pim." + sample_prefix
                                                + (control.algorithm == "ssd"
                                                    ? ".tree_step" + std::to_string(work.tree_step)
                                                      + ".depth" + std::to_string(work.branch_depth)
                                                      + ".branch" + std::to_string(work.branch_index)
                                                    : ".proposal" + std::to_string(work.branch_depth))
                                                + (collapse_layers ? ".macro" : ".layer" + std::to_string(layer_id))
                                                + ".op." + op.name;
                        if (aggregate_micro) {
                            uint64_t summed_delay_ns = 0;
                            uint64_t micro_command_count = 0;
                            for (const auto& command_name : micro_profile.ordered_cmd_name) {
                                const auto [delay, count] = micro_profile.cmd_map.at(command_name);
                                summed_delay_ns += static_cast<uint64_t>(delay) * count * tile_count * layer_multiplier;
                                micro_command_count += static_cast<uint64_t>(count) * tile_count * layer_multiplier;
                            }
                            // Bridge adds 0.5 ns to each PIM command.  Preserve
                            // that accumulated overhead to within 0.5 ns while
                            // submitting one macro command.
                            const double scaled_total_ns =
                                (static_cast<double>(summed_delay_ns)
                                 + 0.5 * micro_command_count)
                                * micro_profile.channel_latency_scale;
                            // PIMWrapper adds one final 0.5 ns completion
                            // overhead for the macro itself.
                            const uint64_t macro_delay_ns = static_cast<uint64_t>(std::max(
                                0.0, std::round(scaled_total_ns - 0.5)));
                            tlm_generic_payload* trans = generate_trans(
                                TraceType::PIM, HOST, PIM, static_cast<unsigned int>(macro_delay_ns),
                                0, 1, layer, "macro(" + std::to_string(layer_multiplier) + "x" + op.name + ")");
                            trans->set_last(++transaction_index == total_transactions);
                            pending_q.push_back(trans);
                        } else {
                            for (uint32_t tile = 0; tile < tile_count; ++tile) {
                                for (const auto& command_name : micro_profile.ordered_cmd_name) {
                                    const auto [delay, count] = micro_profile.cmd_map.at(command_name);
                                    const auto scaled_delay = static_cast<unsigned int>(std::max(
                                        1.0, std::round(delay * micro_profile.channel_latency_scale)));
                                    for (uint32_t burst = 0; burst < count; ++burst) {
                                        tlm_generic_payload* trans = generate_trans(
                                            TraceType::PIM, HOST, PIM, scaled_delay, burst, count,
                                            layer, command_name);
                                        trans->set_last(++transaction_index == total_transactions);
                                        pending_q.push_back(trans);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        } else {
            const auto& profile = profile_it->second;
            const uint32_t tile_count = tile_it->second;
            uint64_t per_proposal_bursts = 0;
            for (const auto& command_name : profile.ordered_cmd_name) {
                per_proposal_bursts += profile.cmd_map.at(command_name).count;
            }
            const uint64_t total_bursts = draft_work_items.size() * tile_count * per_proposal_bursts;
            uint64_t burst_index = 0;
            for (const auto& work : draft_work_items) {
                for (uint32_t tile = 0; tile < tile_count; ++tile) {
                    for (const auto& command_name : profile.ordered_cmd_name) {
                        const auto [delay, count] = profile.cmd_map.at(command_name);
                        const auto scaled_delay = static_cast<unsigned int>(std::max(
                            1.0, std::round(delay * profile.channel_latency_scale)));
                        for (uint32_t burst = 0; burst < count; ++burst) {
                            tlm_generic_payload* trans = generate_trans(
                                TraceType::PIM, HOST, PIM, scaled_delay, burst, count,
                            spec.draft_profile_layer + (control.algorithm == "ssd"
                                ? ".tree_step" + std::to_string(work.tree_step)
                                  + ".depth" + std::to_string(work.branch_depth)
                                  + ".branch" + std::to_string(work.branch_index)
                                : ".proposal" + std::to_string(work.branch_depth)), command_name);
                            trans->set_last(++burst_index == total_bursts);
                            pending_q.push_back(trans);
                        }
                    }
                }
            }
        }

        // The next stage requires both the single NPU core and the draft
        // precompute result.  This is a pipeline-stage completion for SSD,
        // not a same-round proposal/verification dependency.
        wait(pim_trace_done_ev & npu_trace_done_ev);
        logger->update_end(barrier_layer, HOST);
        std::cout << (ssd_pipeline ? "[SSD][Pipeline][StageComplete]" : "[Speculative][Barrier]")
                  << " iter=" << control.iteration
                  << " decision=" << control.decision
                  << " accepted=" << control.accepted_tokens;
        if (control.rollback_to >= 0) std::cout << " rollback_to=" << control.rollback_to;
        std::cout << " cycle=" << current_cycle() << "\n";
    }
}

void Host::handle_memory_trace(const std::shared_ptr<TraceGenerator::MemoryTrace>& trace)
{
    const std::string& layer = trace->layer;

    /* Handle dependencies */
    if (trace->type == TraceType::READ) {
        /* Ensure data is ready before reading */
        wait_sync_signal(trace->id, trace->dst);
    }
    else {
        /* Wait for compute completion before writing */
        if (!is_layer_compute_done[layer] && (layer.find("RoPE_") == std::string::npos)) {
            wait_host_compute(layer);
        }
    }
    logger->update_start(layer, HOST);

    /* Split request into multiple transactions */
    unsigned int trans_num = (trace->size + dram_req_bytes - 1) / dram_req_bytes;

    for (size_t i = 0; i < trans_num; i++) {
        tlm_generic_payload* trans = generate_trans(
            trace->type, /* Read/Write  */
            trace->src,  /* Source      */
            trace->dst,  /* Destination */
            trace->address + (i * dram_req_bytes),
            i,           /* Burst ID    */
            trans_num,   /* Burst size  */
            layer
        );
        trans->set_head(i == 0);
        if (trace->type == TraceType::WRITE) {
            trans->set_last(i == (trans_num - 1));
        }

        pending_q.push_back(trans);
    }

    /* Wait until all read responses are received */
    if (trace->type == TraceType::READ && trace->src == HOST) {
        wait_read_responses(trans_num, layer, false);
    }
}

void Host::handle_compute_trace(const std::shared_ptr<TraceGenerator::ComputeTrace>& trace)
{
    const std::string& layer = trace->layer;

    if (trace->device == HOST) {
        logger->update_start(layer, HOST);

        /* Simulate computation delay */
        const auto op_type = get_op_type(layer);
        unsigned int delay = host_profile[op_type];
        std::cout << "[Host][COMPUTE] " << layer << " (" << delay << " ns)" << std::endl;
        wait(delay, SC_NS);

        /* Compute completion */
        is_layer_compute_done[layer] = true;
        ev_host_compute_done.notify();

        logger->update_end(layer, HOST);
    }
    else {
        /* NPU: Init compute, MCU: Macro PIM command */
        tlm_generic_payload* trans = m_mm.allocate();
        trans->acquire();
        trans->set_command(TLM_COMPUTE_COMMAND);
        trans->set_dst_id(trace->device);
        trans->set_layer(layer);
        pending_q.push_back(trans);
    }
}

void Host::execute_addertree(const std::string& layer, const std::string& cmd, unsigned int next)
{
    /* Wait for PIM computation end */
    auto it = pim_sync_map.find(layer);
    if (it == pim_sync_map.end()) {
        throw std::runtime_error("Missing PIM completion mapping for layer: " + layer);
    }

    auto [sync_id, size, address] = it->second;
    wait_sync_signal(sync_id, PIM);
    unsigned int read_trans_num = (size * 16) / dram_req_bytes; // 16 FP16 elements per dim

    /* Read input from PIM */
    for (int i = 0; i < read_trans_num; i++) {
        tlm_generic_payload *read_trans = generate_trans(
            TraceType::READ,
            HOST, /* Source */
            PIM, /* Destination */
            0, /* Profiled delay */
            i, /* Burst ID */
            read_trans_num, /* Burst size */
            layer,
            cmd
        );
        pending_q.push_back(read_trans);
    }

    /* Wait until all read responses are received */
    wait_read_responses(read_trans_num, layer, true);

    /* Simulate AdderTree computation delay */
    unsigned int delay = host_profile["addertree"];
    std::cout << "[Host][AdderTree][COMPUTE] " << layer << "\n";
    wait(delay, SC_NS);

    unsigned int write_trans_num = (size + dram_req_bytes - 1) / dram_req_bytes;

    /* Write result back to PIM */
    for (int i = 0; i < write_trans_num; i++) {
        tlm_generic_payload *write_trans = generate_trans(
            TraceType::WRITE,
            HOST, /* Source */
            next, /* Destination */
            0, /* Profiled delay */
            i, /* Burst ID */
            write_trans_num, /* Burst ID */
            layer,
            cmd
        );
        write_trans->set_last(i == write_trans_num - 1);
        pending_q.push_back(write_trans);
    }
}

void Host::handle_pim_trace(const std::shared_ptr<TraceGenerator::PimTrace>& trace)
{
    const std::string& layer = trace->layer;
    const std::string& micro_cmd = trace->cmd;
    unsigned int next_memory = trace->next;

    if (micro_cmd == "addertree") {
        execute_addertree(layer, micro_cmd, next_memory);
        return;
    }

    /* Handle PIM sub-ops */
    const auto& profile = pim_profile[layer];
    unsigned int delay = profile.cmd_map.at(micro_cmd).delay;
    unsigned int count = profile.cmd_map.at(micro_cmd).count;
    size_t num_cmds = profile.cmd_map.size();
    unsigned int tile_count = tile_cnt_by_layer[layer];
    cmd_id++;

    if (micro_cmd == "mac") {
        delay = delay * 0.6;
    }

    /* Wait for input dependencies to be ready */
    auto range = pim_wait_map.equal_range(layer);
    for (auto it = range.first; it != range.second; ++it) {
        unsigned int id = std::get<0>(it->second);
        if (!pim_sync_obj.check_signal(id)) {
            wait(*pim_sync_obj.get_event(id));
        }
    }

    if (layer.find("RoPE") != std::string::npos) {
        if (layer != previous_layer) {
            if (is_pim_busy) {
                wait(pim_compute_done_ev);  /* PIM이 이전 RoPE 연산을 마칠 때까지 기다림 */
            }
            is_pim_busy = true;
        }
        previous_layer = layer;
    }

    logger->update_start(layer, HOST);

    /* Generate micro PIM transactions */
    for (int i = 0; i < count; i++) {
        tlm_generic_payload *pim_trans = generate_trans(
            TraceType::PIM,
            HOST, /* Source */
            PIM, /* Destination */
            delay, /* Profiled delay */
            i, /* Burst ID */
            count, /* Burst size */
            layer,
            micro_cmd
        );

        pim_trans->set_last((cmd_id == num_cmds*tile_count) && (i == count - 1));
        if (pim_trans->is_last()) { cmd_id = 0; }
        pending_q.push_back(pim_trans);
    }
}

tlm_generic_payload* Host::generate_trans(TraceType type, unsigned int src, unsigned int dst, unsigned int addr,
                                        unsigned int burst_id, unsigned int burst_size,
                                        const std::string& layer, const std::string& micro_cmd)
{
    tlm_generic_payload* trans = m_mm.allocate();
    trans->acquire();
    trans->set_pim_cmd(micro_cmd);
    trans->set_address(addr);
    trans->set_bst_size(burst_size);
    trans->set_bst_id(burst_id);
    trans->set_src_id(src);
    trans->set_dst_id(dst);
    trans->set_layer(layer);
    trans->set_command((type == TraceType::PIM) ? TLM_PIM_COMMAND :
        (type == TraceType::WRITE) ? TLM_WRITE_COMMAND : TLM_READ_COMMAND);
    return trans;
}

void Host::wait_sync_signal(unsigned int id, unsigned int dst)
{
    SyncObject& sync_obj = (dst == DRAM) ? dram_sync_obj : pim_sync_obj;
    if (!sync_obj.check_signal(id)) {
        wait(*sync_obj.get_event(id));
    }
}

void Host::wait_host_compute(const std::string& layer)
{
    if (!is_layer_compute_done[layer]) {
        wait(ev_host_compute_done);
    }
}

void Host::wait_read_responses(unsigned int trans_num, const std::string& layer, bool addertree)
{
    unsigned int response_received = 0;

    while (response_received < trans_num) {
        while (!read_rsp_q.empty()) {
            read_rsp_q.pop_front();
            response_received++;
            if (addertree) {
                std::cout << "[Host][AdderTree][READ RESPONSE] " << layer << ", " << response_received << "/" << trans_num << std::endl;
            } else {
                std::cout << "[Host][READ RESPONSE] " << layer << ", " << response_received << "/" << trans_num << std::endl;
            }
        }

        if (response_received < trans_num) {
            wait(ev_read_rsp_arrived);
        }
    }
}

tlm_sync_enum Host::nb_transport_bw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void Host::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    read_rsp_q.push_back(&trans);
    ev_read_rsp_arrived.notify();
}

const std::string& Host::get_op_type(const std::string& layer)
{
    static const std::string SOFTMAX = "softmax";
    static const std::string RMSNORM = "rmsnorm";
    static const std::string CONN = "residual";
    static const std::string GELU = "gelu";
    if (layer.find("Softmax") != std::string::npos) return SOFTMAX;
    else if (layer.find("RMSNorm") != std::string::npos) return RMSNORM;
    else if (layer == "gelu") return GELU;
    return CONN;
}

void Host::print_log(const tlm_generic_payload* trans)
{
    const std::string& layer = trans->get_layer();
    uint32_t burst_id = trans->get_bst_id();
    uint32_t burst_count = trans->get_bst_size();

    if (trans->get_command() == TLM_READ_COMMAND) {
        if (trans->get_dst_id() == DRAM ) {
            std::cout << "[Host->DRAM][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << ", cycle: " << current_cycle() << "\n";
        } else if (trans->get_dst_id() == PIM) {
            if (trans->get_pim_cmd() == "addertree") {
                std::cout << "[Host->PIM][AdderTree][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            } else {
                std::cout << "[Host->PIM][READ] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            }
        }
    } else if (trans->get_command() == TLM_WRITE_COMMAND) {
        if (trans->get_dst_id() == DRAM ) {
            std::cout << "[Host->DRAM][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << ", cycle: " << current_cycle() << "\n";
        } else if (trans->get_dst_id() == PIM) {
            if (trans->get_pim_cmd() == "addertree") {
                std::cout << "[Host->PIM][AdderTree][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            } else {
                std::cout << "[Host->PIM][WRITE] " << layer << ", " << burst_id+1 << "/" << burst_count << "\n";
            }
        }
    } else {
        std::cout << "[Host->PIM][COMPUTE] " << layer << " " << trans->get_pim_cmd() << ", " << burst_id+1 << "/" << burst_count << "\n";
    }
}
