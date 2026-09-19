#include "pim_wrapper.h"

#include <algorithm>
#include <cctype>

namespace {

std::string episode_id_from_draft_layer(const std::string& layer)
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

PIMWrapper::PIMWrapper(sc_module_name name, Logger* logger, PerformanceMetrics* metrics)
: slave("slave"), clock("clock"), logger(logger), metrics(metrics), peq(this, &PIMWrapper::peq_cb)
{
	if (cfgs.get_memory_structure() == "partitioned") {
		SC_THREAD(simulate_partitioned_pim);
	} else if (cfgs.get_memory_structure() == "unified") {
		SC_THREAD(simulate_unified_pim);
	}

	SC_METHOD(clock_posedge);
    sensitive << clock.pos();
	dont_initialize();

	SC_METHOD(clock_negedge);
    sensitive << clock.neg();
	dont_initialize();

	slave.register_nb_transport_fw(this, &PIMWrapper::nb_transport_fw);

	mode_change_delay = cfgs.get_mode_change_delay();
    pim_sync_map = cfgs.get_pim_sync_map();
	host_sync_map = cfgs.get_host_sync_map();

	init_pim_dram();
	metrics->configure(PerformanceMetrics::Device::PIM, cfgs.get_pim_channels(),
		cfgs.fast_layer_mode_enabled() ? "logical_balanced_pim_model" : "physical_address_mapped");
}

PIMWrapper::~PIMWrapper()
{
	if (bridge) {
		if (!cfgs.fast_layer_mode_enabled()) {
			bridge->print_stats();
		}
		delete bridge;
	}
	std::cout << "Mode_change_count:" << mode_change_count << std::endl;
	std::cout << "Pending_count:" << pending_count << std::endl;
	if (cfgs.get_memory_structure() == "unified") {
		std::cout << "[UnifiedContention] target_memory_wait_ns="
			<< static_cast<int64_t>(target_memory_wait_total.to_seconds() * 1e9)
			<< " (" << target_memory_wait_count << " intervals), draft_compute_wait_ns="
			<< static_cast<int64_t>(draft_compute_wait_total.to_seconds() * 1e9)
			<< " (" << draft_compute_wait_count << " intervals)\n";
	}
}

void PIMWrapper::init_pim_dram()
{
	std::string type = cfgs.get_pim_type();
	std::string config = cfgs.get_pim_config();
	std::string preset = cfgs.get_pim_preset();
	uint32_t channel_num = cfgs.get_pim_channels();

	std::string path(DRAMSIM3_PATH);
	std::string cfg_path = path + "configs/" + type + "/" + config + "_" + preset
								+ "_ch" + std::to_string(channel_num) + ".ini";
	bridge = new dramsim3::Bridge(cfg_path.c_str(), path.c_str());
	bridge->set_stats_prefix(path + "hsim_pim_stats");
	if (cfgs.fast_layer_mode_enabled()
		&& cfgs.get_memory_structure() == "unified") {
		fast_dram_model = std::make_unique<DramMacroModel>(
			"unified-pim-memory", cfg_path, path, cfgs.get_pim_freq(),
			cfgs.get_dram_req_size(), cfgs.get_pim_channels(), cfgs.get_synthetic_npu_info());
	}
}

void PIMWrapper::simulate_partitioned_pim()
{
	double period = cfgs.fast_layer_mode_enabled() ? 1000.0 : 1/cfgs.get_pim_freq();

	while (1) {
		complete_fast_memory_requests();
		complete_fast_compute_requests();
		/* Check completed transactions from bridge */
        if (auto* completed_trans = bridge->getCompletedCommand()) {        
			if (completed_trans->get_command() == TLM_PIM_COMMAND) {
				auto done_it = detailed_compute_done_time.find(completed_trans);
				metrics->request_completed(PerformanceMetrics::Device::PIM, completed_trans,
					done_it == detailed_compute_done_time.end() ? sc_time_stamp() : done_it->second);
				if (done_it != detailed_compute_done_time.end()) detailed_compute_done_time.erase(done_it);
				compute_completed_queue.push_back(completed_trans);
			} else {
				metrics->request_completed(PerformanceMetrics::Device::PIM, completed_trans,
					sc_time_stamp());
				memory_completed_queue.push_back(completed_trans);	
			}
		}

		/* Process pending transactions */
		if (!pending_queue.empty()) {
			auto* trans = pending_queue.front();
			pending_queue.pop_front();
		
			if (trans->get_command() == TLM_PIM_COMMAND) {
				/* Send compute command to PIM DRAM */
				send_transaction(trans, true);
			} else {
				/* Send memory command to PIM DRAM */
				send_transaction(trans, false);
			}
		}

		// Fast-layer macro requests use the local completion queues, not
		// DRAMsim3.  Ticking DRAMsim3 at every coarse 1-us polling interval
		// dominates host runtime without changing simulated timing.
		if (!cfgs.fast_layer_mode_enabled()) bridge->ClockTick();

		/* Check termination condition */
		if (active_host == 0 && active_core == 0) {
			active_dram = 0;
			break;
		}

		/* Fast macros have deterministic completion times.  Sleeping until the
		 * next one avoids polling every microsecond while a long NPU transfer
		 * is in flight; a newly queued request wakes this thread immediately. */
		if (cfgs.fast_layer_mode_enabled() && pending_queue.empty()) {
			sc_time next = sc_time_stamp() + sc_time(1, SC_MS);
			if (!fast_memory_completed_queue.empty())
				next = std::min(next, fast_memory_completed_queue.front().first);
			if (!fast_compute_completed_queue.empty())
				next = std::min(next, fast_compute_completed_queue.front().first);
			wait(next - sc_time_stamp(), pending_event);
		} else {
			wait(period, SC_NS);
		}
	}
}

void PIMWrapper::simulate_unified_pim()
{
	double period = cfgs.fast_layer_mode_enabled() ? 1000.0 : 1/cfgs.get_pim_freq();

	while (1) {
		bool blocked_fast_request = false;
		complete_fast_memory_requests();
		complete_fast_compute_requests();
		/* Check completed memory transactions */
        if (auto* completed_trans = bridge->getCompletedCommand()) {
			if (completed_trans->get_command() == TLM_PIM_COMMAND) {
				auto done_it = detailed_compute_done_time.find(completed_trans);
				metrics->request_completed(PerformanceMetrics::Device::PIM, completed_trans,
					done_it == detailed_compute_done_time.end() ? sc_time_stamp() : done_it->second);
				if (done_it != detailed_compute_done_time.end()) detailed_compute_done_time.erase(done_it);
				compute_completed_queue.push_back(completed_trans);
			} else {
				metrics->request_completed(PerformanceMetrics::Device::PIM, completed_trans,
					sc_time_stamp());
				memory_completed_queue.push_back(completed_trans);
			}
        }

		/* Process pending transactions */
		if (!pending_queue.empty()) {
			auto* trans = pending_queue.front();
            bool process = (trans->get_command() == TLM_PIM_COMMAND) ? !memory_busy : !compute_busy;

			if (trans->get_command() == TLM_PIM_COMMAND) { /* PIM Compute command */				
				if (process) {
					/* Send compute command to PIM DRAM */
					pending_queue.pop_front();
					finish_contention_wait(trans);
					send_transaction(trans, true);
				} else {
					pending_count++;
					contention_wait_start.try_emplace(trans, sc_time_stamp());
					blocked_fast_request = cfgs.fast_layer_mode_enabled();
					/* Wait for completion */
				}
			} else { /* PIM DRAM Read/Write command */
				if (process) {
					/* Send memory command to PIM DRAM */
					pending_queue.pop_front();
					finish_contention_wait(trans);
					send_transaction(trans, false);
				} else {
					pending_count++;
					contention_wait_start.try_emplace(trans, sc_time_stamp());
					blocked_fast_request = cfgs.fast_layer_mode_enabled();
					/* Wait for completion */
				}
			}
		}

		// See partitioned path: macro-mode requests bypass DRAMsim3.
		if (!cfgs.fast_layer_mode_enabled()) bridge->ClockTick();

		/* Check termination condition */
		if (active_host == 0 && active_core == 0) {
			active_dram = 0;
			break;
		}

		if (cfgs.fast_layer_mode_enabled()
			&& (pending_queue.empty() || blocked_fast_request)) {
			sc_time next = sc_time_stamp() + sc_time(1, SC_MS);
			if (!fast_memory_completed_queue.empty())
				next = std::min(next, fast_memory_completed_queue.front().first);
			if (!fast_compute_completed_queue.empty())
				next = std::min(next, fast_compute_completed_queue.front().first);
			wait(next - sc_time_stamp(), pending_event | resource_available_event);
		} else {
			wait(period, SC_NS);
		}
	}
}

void PIMWrapper::finish_contention_wait(tlm_generic_payload* trans)
{
	auto it = contention_wait_start.find(trans);
	if (it == contention_wait_start.end()) return;
	const sc_time start = it->second;
	const sc_time end = sc_time_stamp();
	contention_wait_start.erase(it);
	if (end <= start) return;

	const bool draft_compute = trans->get_command() == TLM_PIM_COMMAND;
	const bool target_memory = !draft_compute && trans->get_src_id() == NPU;
	if (!draft_compute && !target_memory) return;

	const std::string layer = draft_compute
		? "unified_contention.draft_compute_wait." + std::to_string(draft_compute_wait_count++)
		: "unified_contention.target_memory_wait." + std::to_string(target_memory_wait_count++);
	logger->update_start_at(layer, PIM, static_cast<int64_t>(start.to_seconds() * 1e9));
	logger->update_end_at(layer, PIM, static_cast<int64_t>(end.to_seconds() * 1e9));
	if (draft_compute) draft_compute_wait_total += end - start;
	else target_memory_wait_total += end - start;
	metrics->record_pim_resource_stall(start, end, draft_compute);
}

void PIMWrapper::send_transaction(tlm::tlm_generic_payload* trans, bool is_compute)
{
    if (is_compute) {
        process_mode_transition(PimMode::COMPUTE);
		compute_busy = true;
		compute_busy_count++;
		if (cfgs.fast_layer_mode_enabled()
			&& trans->get_pim_cmd().rfind("macro(", 0) == 0) {
			const sc_time actual_start = std::max(sc_time_stamp(), fast_compute_available);
			const sc_time done_time = actual_start
				+ sc_time(static_cast<double>(trans->get_address()), SC_NS)
				+ sc_time(0.5, SC_NS);
			fast_compute_completed_queue.emplace_back(done_time, trans);
			fast_compute_available = done_time;
			metrics->request_started(PerformanceMetrics::Device::PIM, trans, actual_start);
			// A macro sits in the compute queue until every earlier macro
			// drains.  Record when it actually starts computing, not when it
			// was handed to the queue, or every bar in the timeline would span
			// its whole queueing delay and they would all appear to overlap.
			logger->update_start_at(trans->get_layer(), PIM,
				static_cast<int64_t>(actual_start.to_double() / 1000));
			return;
		}
		bridge->sendCommand(*trans);
		const sc_time actual_start = bridge->getLastPimComputeStartTime();
		detailed_compute_done_time[trans] = bridge->getLastPimComputeDoneTime();
		metrics->request_started(PerformanceMetrics::Device::PIM, trans, actual_start);
		logger->update_start_at(trans->get_layer(), PIM,
			static_cast<int64_t>(actual_start.to_double() / 1000));
	} else {
        process_mode_transition(PimMode::MEMORY);
		memory_busy = true;
		memory_busy_count++;
		if (trans->get_pim_cmd() == "fast_memory") {
			const uint64_t duration_ns = fast_dram_model->estimate_ns(
				trans->get_command(), DramMacroModel::macro_bytes(*trans));
			const sc_time actual_start = std::max(sc_time_stamp(), fast_memory_available);
			const sc_time done_time = actual_start
				+ sc_time(static_cast<double>(duration_ns), SC_NS);
			fast_memory_available = done_time;
			logger->update_start_at(trans->get_layer(), PIM,
				static_cast<int64_t>(actual_start.to_double() / 1000));
			metrics->request_started(PerformanceMetrics::Device::PIM, trans, actual_start);
			fast_memory_completed_queue.emplace_back(
				done_time, trans);
			return;
		}
		metrics->request_started(PerformanceMetrics::Device::PIM, trans, sc_time_stamp(),
			bridge->getChannel(trans->get_address()));
        bridge->sendCommand(*trans);
    }
}

void PIMWrapper::complete_fast_memory_requests()
{
	while (!fast_memory_completed_queue.empty()
		&& fast_memory_completed_queue.front().first <= sc_time_stamp()) {
		const auto& [done_time, trans] = fast_memory_completed_queue.front();
		logger->update_end_at(trans->get_layer(), PIM,
			static_cast<int64_t>(done_time.to_double() / 1000));
		metrics->request_completed(PerformanceMetrics::Device::PIM, trans, done_time);
		memory_completed_queue.push_back(trans);
		fast_memory_completed_queue.pop_front();
	}
}

void PIMWrapper::complete_fast_compute_requests()
{
	while (!fast_compute_completed_queue.empty()
		&& fast_compute_completed_queue.front().first <= sc_time_stamp()) {
		const auto& [done_time, trans] = fast_compute_completed_queue.front();
		// Pair the exact completion time with the exact start recorded in
		// send_transaction(), so a bar's width is the modeled delay rather than
		// the delay rounded up to the next PIM tick.
		logger->update_end_at(trans->get_layer(), PIM,
			static_cast<int64_t>(done_time.to_double() / 1000));
		metrics->request_completed(PerformanceMetrics::Device::PIM, trans, done_time);
		if (trans->get_layer().rfind("draft_pim.", 0) == 0) {
			++fast_macro_completed;
			if (fast_macro_completed % 64 == 0 || trans->is_last()) {
				std::cout << "[SyntheticPIM][FAST] completed macros="
					<< fast_macro_completed << " sim_ns="
					<< static_cast<int64_t>(done_time.to_seconds() * 1e9) << "\n";
			}
		}
		compute_completed_queue.push_back(trans);
		fast_compute_completed_queue.pop_front();
	}
}

void PIMWrapper::process_mode_transition(PimMode target_mode)
{
	if (pim_mode != target_mode) {
		mode_change_count++;
		unsigned int  delay = (target_mode == PimMode::MEMORY)
			? std::get<0>(mode_change_delay)
			: std::get<1>(mode_change_delay);

		wait(sc_time(delay, SC_NS));
		pim_mode = target_mode;
	}		
}

void PIMWrapper::clock_posedge()
{
	if (memory_completed_queue.empty()) return;

	auto* completed_trans = memory_completed_queue.front();
	memory_completed_queue.pop_front();

	bool is_read =
		(completed_trans->get_command() == TLM_READ_COMMAND) ||
		(completed_trans->get_src_id() == NPU);

	if (is_read) {
		/* Read response to Host, Read/Write response to NPU */
		unsigned int dst_id = (completed_trans->get_src_id() == MCU) ? 1 : 0;
		send_response(completed_trans, dst_id);
	}else if (completed_trans->get_command()==TLM_WRITE_COMMAND) {
		/* Notify Write completion */
		complete_write_request(completed_trans);
	}
	if (--memory_busy_count == 0) {
		memory_busy = false;
		resource_available_event.notify(SC_ZERO_TIME);
	}
}

void PIMWrapper::clock_negedge()
{	
	if (compute_completed_queue.empty()) return ;

	auto* trans = compute_completed_queue.front();
	compute_completed_queue.pop_front();

	const std::string& layer = trans->get_layer();
	if (layer.rfind("draft_pim.", 0) == 0) {
		std::cout << "[SyntheticPIM] completed " << layer
				  << " cmd=" << trans->get_pim_cmd()
				  << " cycle=" << current_cycle() << "\n";
		// Fast macros already recorded their exact end in
		// complete_fast_compute_requests(); do not overwrite it with the tick
		// on which this queue happened to be drained.
		if (!cfgs.fast_layer_mode_enabled()) {
			logger->update_end(layer, PIM);
		}
	}
	if (trans->is_last()) {
		if (layer.rfind("draft_pim.", 0) == 0) {
			metrics->complete_parallel_pim_episode(episode_id_from_draft_layer(layer), sc_time_stamp());
		}
		if (layer.find("RoPE") != std::string::npos || layer == "Mul1") {
			/* Computation end signal */
			auto src = trans->get_src_id();
			if (src == HOST) logger->update_end(layer, HOST);
			else if (src == MCU) logger->update_end(layer, MCU);
			is_pim_busy = false;
			pim_compute_done_ev.notify();
		}
		// Synthetic speculative layers are completed through pim_trace_done_ev;
		// they intentionally have no legacy layer-to-sync-map entry.
		if (layer.rfind("draft_pim.", 0) != 0) {
			signal_sync(pim_sync_map, layer);
		}
		if (layer.rfind("draft_pim.", 0) != 0) {
			logger->update_end(layer, PIM);
		}
		pim_trace_done_ev.notify(SC_ZERO_TIME);
	}

	if (--compute_busy_count == 0) {
		compute_busy = false;
		resource_available_event.notify(SC_ZERO_TIME);
	}
}

void PIMWrapper::send_response(tlm::tlm_generic_payload* trans, unsigned int dst_id)
{	
	tlm_phase phase = BEGIN_RESP;
	tlm_sync_enum reply = slave[dst_id]->nb_transport_bw(*trans, phase, t);
	assert(reply == TLM_UPDATED);
}

void PIMWrapper::complete_write_request(tlm::tlm_generic_payload* trans)
{	
	const std::string& layer = trans->get_layer();
	BaseMap* base_map = (trans->get_pim_cmd() == "addertree") ? &pim_sync_map : &host_sync_map;

	if (trans->is_last()) {
		signal_sync(*base_map, layer);
		if (trans->get_src_id() == HOST) {
			logger->update_end(layer, HOST);
		}
		else if (trans->get_src_id() == MCU) {
			logger->update_end(layer, MCU);
		}
	}
	trans->release();
}

tlm_sync_enum PIMWrapper::nb_transport_fw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
	peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void PIMWrapper::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
	pending_queue.push_back(&trans);
	const bool compute = trans.get_command() == TLM_PIM_COMMAND;
	const uint64_t bytes = compute ? 0 : (DramMacroModel::is_macro_request(trans)
		? DramMacroModel::macro_bytes(trans) : trans.get_data_length());
	metrics->request_arrived(PerformanceMetrics::Device::PIM, &trans, bytes, compute);
	pending_event.notify(SC_ZERO_TIME);
}

void PIMWrapper::signal_sync(BaseMap& map, const std::string& layer)
{
	auto it = map.find(layer);
	if (it != map.end()) {
		unsigned int sync_id = std::get<0>(it->second);
		std::cout << "[PIM][SIGNAL] " << layer << " (ID:" << sync_id << "), cycle:" << current_cycle() << "\n";
		pim_sync_obj.signal(sync_id);
	} else {
		std::cerr << "[PIM][ERROR] Key[" << layer << "] not found in the map!\n";
	}
}
