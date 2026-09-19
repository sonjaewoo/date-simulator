#include "mem_wrapper.h"

MEMWrapper::MEMWrapper(sc_module_name name, Logger* logger, PerformanceMetrics* metrics)
: slave("slave"), clock("clock"), peq(this, &MEMWrapper::peq_cb), logger(logger), metrics(metrics)
{
	SC_THREAD(simulate_dram);

	SC_METHOD(clock_posedge);
    sensitive << clock.pos();
	dont_initialize();

	SC_METHOD(clock_negedge);
    sensitive << clock.neg();
	dont_initialize();
	
	slave.register_nb_transport_fw(this, &MEMWrapper::nb_transport_fw);

	dram_req_size = cfgs.get_dram_req_size();
    host_sync_map = cfgs.get_host_sync_map();
    pim_sync_map = cfgs.get_pim_sync_map();
	init_dram();
	metrics->configure(PerformanceMetrics::Device::DRAM, cfgs.get_dram_channels(),
		cfgs.fast_layer_mode_enabled() ? "logical_balanced_macro" : "physical_address_mapped");
	if (cfgs.fast_layer_mode_enabled()) {
		std::string path(DRAMSIM3_PATH);
		std::string cfg_path = path + "configs/" + cfgs.get_dram_type() + "/"
			+ cfgs.get_dram_config() + "_" + cfgs.get_dram_preset() + "_ch"
			+ std::to_string(cfgs.get_dram_channels()) + ".ini";
		fast_dram_model = std::make_unique<DramMacroModel>(
			"partitioned-dram", cfg_path, path, cfgs.get_dram_freq(),
			dram_req_size, cfgs.get_dram_channels(), cfgs.get_synthetic_npu_info());
	}
}

MEMWrapper::~MEMWrapper()
{
	if (dramsim3_bridge) {
		if (!cfgs.fast_layer_mode_enabled()) {
			dramsim3_bridge->print_stats();
		}
		delete dramsim3_bridge;
	}	
}

void MEMWrapper::init_dram()
{
	std::string type = cfgs.get_dram_type();
	std::string preset = cfgs.get_dram_preset();
	std::string config = cfgs.get_dram_config();
	uint32_t channel_num = cfgs.get_dram_channels();


	std::string path(DRAMSIM3_PATH);
	std::string cfg_path = path + "configs/" + type + "/" + config + "_" + preset
								+ "_ch" + std::to_string(channel_num) + ".ini";
	dramsim3_bridge = new dramsim3::Bridge(cfg_path.c_str(), path.c_str());
	dramsim3_bridge->set_stats_prefix(path + "hsim_dram_stats");
}

void MEMWrapper::clock_posedge() {
    if (!fw_queue.empty()) {
		tlm_generic_payload *trans = fw_queue.front();
		fw_queue.pop_front();
		mem_request.push_back(trans);
        fast_memory_event.notify(SC_ZERO_TIME);
    }
}

void MEMWrapper::clock_negedge() {
	if (!bw_queue.empty())  {
		tlm_generic_payload* trans = bw_queue.front();
		bw_queue.pop_front();

		bool is_read = (trans->get_command() == TLM_READ_COMMAND) ? true : false;
		backward_trans(trans, is_read);
		m_outstanding--;
	}
}

tlm_sync_enum MEMWrapper::nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t) {
	peq.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void MEMWrapper::peq_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
	fw_queue.push_back(&trans);
	const uint64_t bytes = DramMacroModel::is_macro_request(trans)
		? DramMacroModel::macro_bytes(trans) : trans.get_data_length();
	metrics->request_arrived(PerformanceMetrics::Device::DRAM, &trans, bytes);
}

unsigned int MEMWrapper::find_sync_id(std::string layer)
{
    auto it = host_sync_map.find(layer);
	if (it != host_sync_map.end()) {
		const auto& [id, size, address] = it->second;
		return id;
	} else {
		throw std::runtime_error("[DRAM][ERROR] Address not found in map.");
	}
}

void MEMWrapper::simulate_dram()
{
    double period = cfgs.fast_layer_mode_enabled() ? 1000.0 : 1 / cfgs.get_dram_freq();
	unsigned int sync_id;
	int wack_num = 0;

	while (1) {
		complete_fast_memory_requests();
		/* Get completed request from the DRAM simulator */
		tlm_generic_payload* trans = nullptr;
        trans = dramsim3_bridge->getCompletedCommand();

		if (trans) {
			auto cmd = trans->get_command();
			auto src_id = trans->get_src_id();
			const std::string& layer = trans->get_layer();

			if (cmd == TLM_READ_COMMAND) {
				metrics->request_completed(PerformanceMetrics::Device::DRAM, trans,
					sc_time_stamp());
				bw_queue.push_back(trans);
			} else if (cmd == TLM_WRITE_COMMAND) {
				metrics->request_completed(PerformanceMetrics::Device::DRAM, trans,
					sc_time_stamp());
				if (src_id == HOST && trans->get_pim_cmd() == "addertree") {
					complete_write_request(trans);
					m_outstanding--;
				}				
				/* Signal when HOST write (burst) is done */
				else if (src_id == HOST) {
					wack_num++;
					if (wack_num == (trans->get_bst_size())) {
						wack_num = 0;
						unsigned int sync_id = find_sync_id(layer);
						std::cout << "[DRAM][SIGNAL] " << layer << " (ID:" << sync_id << ")\n";
						dram_sync_obj.signal(sync_id);

						logger->update_end(layer, HOST);
					}
					m_outstanding--;
					trans->release();
				}

				else if (src_id == NPU) {
					bw_queue.push_back(trans);
				}
			}
		}
		
		/* Send command to the DRAM simulator */
		while (!mem_request.empty()) {
			tlm_generic_payload *req_trans = mem_request.front();
			mem_request.pop_front();
			m_outstanding++;
			if (cfgs.fast_layer_mode_enabled()
                && DramMacroModel::is_macro_request(*req_trans)) {
                const uint64_t duration_ns = fast_dram_model->estimate_ns(
                    req_trans->get_command(), DramMacroModel::macro_bytes(*req_trans));
                logger->update_start(req_trans->get_layer(), DRAM);
				metrics->request_started(PerformanceMetrics::Device::DRAM, req_trans,
					sc_time_stamp());
                fast_memory_completed_queue.emplace_back(
                    sc_time_stamp() + sc_time(static_cast<double>(duration_ns), SC_NS), req_trans);
            } else {
				metrics->request_started(PerformanceMetrics::Device::DRAM, req_trans,
					sc_time_stamp(), dramsim3_bridge->getChannel(req_trans->get_address()));
                dramsim3_bridge->sendCommand(*req_trans);
            }
		}

		/* Simulate the DRAM simulator */
		// In fast-layer mode Target memory uses the analytical macro path;
		// no request reaches this bridge, so skip an otherwise expensive tick.
		if (!cfgs.fast_layer_mode_enabled()) dramsim3_bridge->ClockTick();

		/* Check if DRAM simulation can stop */
		if (active_host == 0 && active_core == 0 && m_outstanding == 0) {
			active_dram = 0;
			break;
		}

		/* Fast macro completions are event driven.  This avoids polling every
		 * DRAM cycle while still returning the TLM response at the calibrated
		 * DRAMsim3 service time. */
		if (cfgs.fast_layer_mode_enabled() && mem_request.empty()
			&& !fast_memory_completed_queue.empty()) {
			const sc_time next = fast_memory_completed_queue.front().first;
			wait(next - sc_time_stamp(), fast_memory_event);
		} else {
			/* Wait for the next clock period */
			wait(period, SC_NS);
		}
		total_cycle++;
    }
}

void MEMWrapper::complete_fast_memory_requests()
{
	while (!fast_memory_completed_queue.empty()
		&& fast_memory_completed_queue.front().first <= sc_time_stamp()) {
		auto* trans = fast_memory_completed_queue.front().second;
		logger->update_end(trans->get_layer(), DRAM);
		metrics->request_completed(PerformanceMetrics::Device::DRAM, trans, sc_time_stamp());
		bw_queue.push_back(trans);
		fast_memory_completed_queue.pop_front();
	}
}

void MEMWrapper::backward_trans(tlm_generic_payload* trans, bool read) {
	tlm_phase phase = BEGIN_RESP;
	tlm_sync_enum reply = slave->nb_transport_bw(*trans, phase, t);
	assert(reply == TLM_UPDATED);
}

void MEMWrapper::complete_write_request(tlm::tlm_generic_payload* trans)
{	
	if (trans->is_last()) {
		const std::string& layer = trans->get_layer();
		BaseMap* base_map = (trans->get_pim_cmd() == "addertree") ? &pim_sync_map : &host_sync_map;	

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

void MEMWrapper::signal_sync(BaseMap& map, const std::string& layer)
{
	auto it = map.find(layer);
	if (it != map.end()) {
		unsigned int sync_id = std::get<0>(it->second);
		std::cout << "[PIM][AdderTree][SIGNAL] " << layer << " (ID:" << sync_id << "), cycle:" << current_cycle() << "\n";
		if (layer == "D_V0" || layer == "D_V2" || layer == "D_V4" || layer == "D_V6" ||
			layer == "D_V8" || layer == "D_V10" || layer == "D_V12" || layer == "D_V14") {
			dram_sync_obj.signal(sync_id);
		}
		pim_sync_obj.signal(sync_id);
	} else {
		std::cerr << "[PIM][ERROR] Key[" << layer << "] not found in the map!\n";
	}
}
