#ifndef __HOST_H
#define __HOST_H

#include <iomanip>

#include <utilities/mm.h>
#include <utilities/logger.h>
#include <utilities/common.h>
#include <utilities/sync_object.h>
#include <utilities/configurations.h>
#include <utilities/performance_metrics.h>
#include <components/trace_generator.h>
#include <components/npu_trace_replayer.h>
#include <tlm_utils/multi_passthrough_initiator_socket.h>
#include <tlm_utils/peq_with_cb_and_phase.h>

using namespace tlm;
using namespace sc_core;

/* External global references */
extern Configurations cfgs;
extern SyncObject dram_sync_obj;
extern SyncObject pim_sync_obj;
extern unsigned int active_host;
extern unsigned int active_dram;

class Host: public sc_module {
public:
    SC_HAS_PROCESS(Host);
    Host(sc_module_name name, Logger* logger, PerformanceMetrics* metrics);

    /* TLM interface */
    tlm_utils::peq_with_cb_and_phase<Host> peq;
    tlm_utils::multi_passthrough_initiator_socket<Host> master;
    sc_in<bool> clock;    
    
    /* TLM function */
    void clock_negedge();
    void peq_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    tlm_sync_enum nb_transport_bw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);

    /* Trace handler */
    void process_trace();
    void handle_memory_trace(const std::shared_ptr<TraceGenerator::MemoryTrace>& trace);
    void handle_compute_trace(const std::shared_ptr<TraceGenerator::ComputeTrace>& trace);
    void handle_pim_trace(const std::shared_ptr<TraceGenerator::PimTrace>& trace);
    void handle_speculative_trace(const std::shared_ptr<TraceGenerator::SpeculativeTrace>& trace);
    void execute_addertree(const std::string& layer, const std::string& cmd, unsigned int next);

    /* Sync & Wait */
    void wait_read_responses(unsigned int trans_num, const std::string& layer, bool addertree);
    void wait_sync_signal(unsigned int sync_id, unsigned int dst);
    void wait_host_compute(const std::string& layer);

    tlm_generic_payload* generate_trans(
        TraceType type,
        unsigned int src, unsigned int dst,
        unsigned int addr,
        unsigned int burst_id, unsigned int burst_size,
        const std::string& layer, const std::string& micro_cmd = ""
    );
    
    const std::string& get_op_type(const std::string& layer);
    void print_log(const tlm_generic_payload* trans);
	inline int64_t current_cycle() const { return static_cast<int64_t>(sc_time_stamp().to_double()/1000); }

private:
    mm m_mm;
    sc_time t{SC_ZERO_TIME};

    unsigned int cmd_id{0};
    uint32_t dram_req_bytes{0};
    uint64_t active_cycle{0};

    sc_event ev_host_compute_done;
    sc_event ev_read_rsp_arrived;

    std::unordered_map<std::string, unsigned int> host_profile;
    std::unordered_map<std::string, PIMCmdProfile> pim_profile;

    std::deque<std::shared_ptr<TraceGenerator::Trace>> trace_q;
    std::deque<tlm_generic_payload*> pending_q;
    std::deque<tlm_generic_payload*> read_rsp_q;

    std::map<std::string, bool> is_layer_compute_done;
    std::unordered_map<std::string, unsigned int> tile_cnt_by_layer;

    WaitMap pim_wait_map;
    SyncMap pim_sync_map;
    TraceGenerator trace_generator;

    std::string previous_layer = "";
    Logger* logger;
    PerformanceMetrics* metrics;
};

#endif
