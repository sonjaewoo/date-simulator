#ifndef __ARBITER_H
#define __ARBITER_H

#include <utilities/common.h>
#include <utilities/configurations.h>

#include "tlm_utils/multi_passthrough_initiator_socket.h"
#include "tlm_utils/multi_passthrough_target_socket.h"
#include <tlm_utils/peq_with_cb_and_phase.h>

using namespace tlm;
using namespace sc_core;

extern Configurations cfgs;

class Arbiter: public sc_module {
public:   
    SC_HAS_PROCESS(MEMarb);
    sc_in<bool> clock;
    
    tlm_utils::peq_with_cb_and_phase<Arbiter> peq_fw;
    tlm_utils::peq_with_cb_and_phase<Arbiter> peq_bw;

    tlm_utils::multi_passthrough_initiator_socket<Arbiter> master;
    tlm_utils::multi_passthrough_target_socket<Arbiter> slave;

    Arbiter(sc_module_name name);
    ~Arbiter();

	tlm_sync_enum nb_transport_fw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);
	tlm_sync_enum nb_transport_bw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t);

    void clock_posedge();
    void clock_negedge();

    void peq_fw_cb(tlm_generic_payload& trans, const tlm_phase& phase);
    void peq_bw_cb(tlm_generic_payload& trans, const tlm_phase& phase);

private:
    sc_time t{SC_ZERO_TIME};
    std::deque<std::tuple<tlm_generic_payload*, unsigned int>> fw_queue;
    std::deque<std::tuple<tlm_generic_payload*, unsigned int>> bw_queue;
    std::unordered_map<tlm_generic_payload*, tlm_phase> fw_phase_map;
    std::unordered_map<tlm_generic_payload*, tlm_phase> bw_phase_map;
};

#endif