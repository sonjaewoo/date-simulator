#ifndef NPU_TRACE_REPLAYER_H
#define NPU_TRACE_REPLAYER_H

#include <utilities/configurations.h>
#include <utilities/logger.h>
#include <utilities/mm.h>
#include <utilities/performance_metrics.h>

#include "tlm_utils/peq_with_cb_and_phase.h"
#include "tlm_utils/simple_initiator_socket.h"
#include "tlm_utils/simple_target_socket.h"

using namespace tlm;
using namespace sc_core;

extern Configurations cfgs;
extern unsigned int active_core;
extern sc_event npu_trace_done_ev;

// Coarse target-NPU model used before a MIDAP layer trace is available.
// Each target layer performs: weight READ -> fixed compute -> output WRITE.
class NPUTraceReplayer : public sc_module {
public:
    tlm_utils::simple_initiator_socket<NPUTraceReplayer> master;
    tlm_utils::simple_target_socket<NPUTraceReplayer> slave;
    sc_in<bool> clock;
    tlm_utils::peq_with_cb_and_phase<NPUTraceReplayer> peq_fw;
    tlm_utils::peq_with_cb_and_phase<NPUTraceReplayer> peq_bw;

    NPUTraceReplayer(sc_module_name name, Logger* logger, PerformanceMetrics* metrics);
    SC_HAS_PROCESS(NPUTraceReplayer);

    tlm_sync_enum nb_transport_fw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
    tlm_sync_enum nb_transport_bw(tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);

private:
    void peq_fw_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    void peq_bw_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    void replay_trace();
    void issue_memory(tlm_command command, uint32_t bytes, const std::string& layer);
    void issue_fast_memory(tlm_command command, uint64_t bytes, const std::string& layer);
    void complete_memory_response(tlm_generic_payload& trans);
    uint64_t operation_compute_ns(const ModelOperationInfo& op) const;

    mm m_mm;
    sc_time t{SC_ZERO_TIME};
    sc_event start_event;
    sc_event response_event;
    SyntheticNPUInfo profile;
    ModelLayerProfile layer_profile;
    uint32_t dram_req_bytes{0};
    uint32_t run_id{0};
    uint64_t next_address{0x10000000ULL};
    uint32_t pending_responses{0};
    uint32_t active_context_tokens{1};
    uint32_t active_query_tokens{1};
    uint32_t active_kv_write_tokens{1};
    std::string active_episode_id;
    Logger* logger;
    PerformanceMetrics* metrics;
};

#endif
