#include "bridge.h"

// This is a null function to avoid from linking error with the systemc library
int sc_main(int argc,char **argv) { return 0; };

namespace dramsim3 {

void Bridge::ClockTick() {
    memory_system_.ClockTick();
	for (int c = 0; c < memory_system_.GetChannels(); c++) {
        if (m_incomingQueue[c].size() > 0) {
            bool accepted = false;
            bool write = false;
            uint64_t addr;

            tlm_generic_payload *trans = m_incomingQueue[c].q.front();

            addr = trans->get_address();
            write = trans->is_write();

            accepted = memory_system_.WillAcceptTransaction(addr, write);
            if (accepted) {
                bool ok = false;
                ok = memory_system_.AddTransaction(addr, write);

                if (ok) {
                    m_incomingQueue[c].q.pop_front();

                    if (write)
                        m_writeDone.push_back(make_pair(addr, trans));
                    else
                        m_readDone.push_back(make_pair(addr, trans));
                }
            }
        }
	}

    while (!m_pimComputeQueue.empty()) {
        auto& entry = m_pimComputeQueue.front();
        sc_time done_time = entry.first;
        tlm_generic_payload* trans = entry.second;
        if (sc_time_stamp() >= done_time) {
            m_outgoingQueue.q.push_back(trans);
            m_pimComputeQueue.pop_front();
        } else {
            break;
        }
    }

    clk_++;
    return;
}

void Bridge::ReadCallBack(uint64_t addr)
{
    uint32_t i;

    for (i = 0; i < m_readDone.size(); i++) {
        if (m_readDone[i].first == addr) {
            m_outgoingQueue.q.push_back(m_readDone[i].second);
            m_readDone.erase(m_readDone.begin() + i);
        }
    }
}

void Bridge::WriteCallBack(uint64_t addr)
{
    uint32_t i;

    for (i = 0; i < m_writeDone.size(); i++) {
        if (m_writeDone[i].first == addr) {
            m_outgoingQueue.q.push_back(m_writeDone[i].second);
            m_writeDone.erase(m_writeDone.begin() + i);
        }
    }
}

void Bridge::sendCommand(tlm::tlm_generic_payload &trans)
{   
    if (trans.get_command() == tlm::TLM_PIM_COMMAND) {
        sc_time actual_start = max(sc_time_stamp(), last_compute_done_time);
        sc_time compute_done_time =
            actual_start + sc_time(static_cast<uint64_t>(trans.get_address()), SC_NS) + sc_time(0.5, SC_NS);

        m_pimComputeQueue.push_back(make_pair(compute_done_time, &trans));

        last_compute_start_time = actual_start;
        last_compute_done_time = compute_done_time;
    } else {
        uint64_t addr = trans.get_address();
        int channel = memory_system_.GetChannel(addr);
        m_incomingQueue[channel].q.push_back(&trans);
    }
}

tlm_generic_payload* Bridge::getCompletedCommand(void)
{
    tlm_generic_payload *trans = NULL;

    if (m_outgoingQueue.size()) {
        trans = m_outgoingQueue.q.front();
        m_outgoingQueue.q.pop_front();
    }

    return trans;
}

void Bridge::print_stats()
{
    memory_system_.PrintStats();
}

void Bridge::set_stats_prefix(const std::string& prefix)
{
    memory_system_.SetOutputPrefix(prefix);
}

int Bridge::getChannel(uint64_t address) const
{
    return memory_system_.GetChannel(address);
}

}  // namespace dramsim3
