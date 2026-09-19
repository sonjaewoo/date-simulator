#include "arbiter.h"

Arbiter::Arbiter(sc_module_name name)
: master("master"), slave("slave"), clock("clock"),
peq_fw(this, &Arbiter::peq_fw_cb), peq_bw(this, &Arbiter::peq_bw_cb)
{
    SC_METHOD(clock_posedge);
    sensitive << clock.pos();
    dont_initialize();

	SC_METHOD(clock_negedge);
    sensitive << clock.neg();
    dont_initialize();

    master.register_nb_transport_bw(this, &Arbiter::nb_transport_bw);
    slave.register_nb_transport_fw(this, &Arbiter::nb_transport_fw);
};

Arbiter::~Arbiter() {}

void Arbiter::clock_posedge()
{
    if (!bw_queue.empty()) {
        auto [trans, id] = bw_queue.front();
        bw_queue.pop_front();
        tlm_phase phase = bw_phase_map[trans];
        tlm_sync_enum reply = slave[id]->nb_transport_bw(*trans, phase, t);
        assert(reply == TLM_UPDATED);
    }
}

void Arbiter::clock_negedge()
{
    if (!fw_queue.empty()) {
        auto [trans, id] = fw_queue.front();
        fw_queue.pop_front();
        tlm_phase phase = fw_phase_map[trans];
        tlm_sync_enum reply = master[id]->nb_transport_fw(*trans, phase, t);
        assert(reply == TLM_UPDATED);
    }
}

tlm_sync_enum Arbiter::nb_transport_fw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    /* Forward path: record phase and enqueue */
    if (phase == BEGIN_REQ) {
        fw_phase_map[&trans] = phase;
        peq_fw.notify(trans, phase, SC_ZERO_TIME);
    }
    else if (phase == END_RESP) {
        return TLM_COMPLETED;
    }
	return TLM_UPDATED;
}

tlm_sync_enum Arbiter::nb_transport_bw(int id, tlm_generic_payload& trans, tlm_phase& phase, sc_time& t)
{
    /* Backward path: record phase and enqueue */
    bw_phase_map[&trans] = phase;
    peq_bw.notify(trans, phase, SC_ZERO_TIME);
	return TLM_UPDATED;
}

void Arbiter::peq_fw_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    /* Determine target index */
    unsigned int dst_id = 0;
    if (cfgs.dram_enabled()) {
        switch (trans.get_dst_id()) {
            case DRAM: dst_id = 0; break;
            case PIM:  dst_id = 1; break;
        }
    } else {
        dst_id = 0;
    }
    fw_queue.push_back(std::make_tuple(&trans, dst_id));
}

void Arbiter::peq_bw_cb(tlm_generic_payload& trans, const tlm_phase& phase)
{
    unsigned int src_idx = (trans.get_src_id() == HOST) ? 0 : 1;
    bw_queue.push_back(std::make_tuple(&trans, src_idx));
}
