#ifndef PIM_WRAPPER_H
#define PIM_WRAPPER_H

#include <memory>

#include <utilities/logger.h>
#include <utilities/configurations.h>
#include <components/dram_macro_model.h>
#include <utilities/performance_metrics.h>
#include <components/host.h>
// #include "tlm_utils/simple_target_socket.h"
#include "tlm_utils/multi_passthrough_target_socket.h"
#include <tlm_utils/peq_with_cb_and_phase.h>

#include "bridge.h"

extern Configurations cfgs;
extern bool is_pim_busy;
extern sc_event pim_compute_done_ev;
extern sc_event pim_trace_done_ev;

using namespace tlm;
using namespace sc_core;

enum class PimMode { COMPUTE, MEMORY };

class PIMWrapper: public sc_module
{
public:
    SC_HAS_PROCESS(PIMWrapper);

	sc_in<bool> clock;
	tlm_utils::peq_with_cb_and_phase<PIMWrapper> peq;
	// tlm_utils::simple_target_socket<PIMWrapper> slave;
	tlm_utils::multi_passthrough_target_socket<PIMWrapper> slave;

    PIMWrapper(sc_module_name name, Logger* logger, PerformanceMetrics* metrics);
    ~PIMWrapper();

	void clock_posedge();
    void clock_negedge();
	// tlm_sync_enum nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
	tlm_sync_enum nb_transport_fw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);

	void peq_cb(tlm_generic_payload& trans, const tlm_phase& phase);

    void signal_sync(BaseMap& target_map, const std::string& layer);
	void send_response(tlm::tlm_generic_payload* trans, unsigned int dst_id);
	void complete_write_request(tlm::tlm_generic_payload* trans);
	
	void simulate_partitioned_pim();
	void simulate_unified_pim();
	void init_pim_dram();

	void process_mode_transition(PimMode mode);
	void send_transaction(tlm_generic_payload* trans, bool is_compute);
	void complete_fast_memory_requests();
	void complete_fast_compute_requests();

	inline int64_t current_cycle() const { return static_cast<int64_t>(sc_time_stamp().to_double()/1000); }

private:
	sc_time t{SC_ZERO_TIME};
	Logger* logger;
	PerformanceMetrics* metrics;

	SyncMap pim_sync_map;
    SyncMap host_sync_map;

	std::tuple<unsigned int, unsigned int> mode_change_delay;
	dramsim3::Bridge *bridge;
	std::unique_ptr<DramMacroModel> fast_dram_model;

	std::deque<tlm_generic_payload *> pending_queue;	
	std::deque<tlm_generic_payload *> memory_completed_queue;
	std::deque<tlm_generic_payload *> compute_completed_queue;
	std::deque<std::pair<sc_time, tlm_generic_payload *>> fast_memory_completed_queue;
	std::deque<std::pair<sc_time, tlm_generic_payload *>> fast_compute_completed_queue;
	sc_time fast_compute_available{SC_ZERO_TIME};
	sc_time fast_memory_available{SC_ZERO_TIME};
	uint64_t fast_macro_completed{0};
	std::unordered_map<tlm_generic_payload*, sc_time> detailed_compute_done_time;
	sc_event pending_event;
	sc_event resource_available_event;
	std::unordered_map<tlm_generic_payload*, sc_time> contention_wait_start;
	uint64_t target_memory_wait_count{0};
	uint64_t draft_compute_wait_count{0};
	sc_time target_memory_wait_total{SC_ZERO_TIME};
	sc_time draft_compute_wait_total{SC_ZERO_TIME};
	void finish_contention_wait(tlm_generic_payload* trans);

	PimMode pim_mode{PimMode::MEMORY};

	unsigned int mode_change_count{0};
	unsigned int pending_count{0};

	bool memory_busy{false};
	bool compute_busy{false};

	unsigned int memory_busy_count{0};
    unsigned int compute_busy_count{0};
};
#endif
