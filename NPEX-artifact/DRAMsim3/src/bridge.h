#ifndef PACKAGE_HW_BRIDGE_H
#define PACKAGE_HW_BRIDGE_H

#include <fstream>
#include <functional>
#include <random>
#include <string>

#include "memory_system.h"
#include "cpu.h"

// HSIM Integration
#include <list>
#include <vector>
#include "tlm_utils/simple_initiator_socket.h"

using namespace tlm;
using namespace std;
using namespace sc_core;

namespace dramsim3 {

#define MAX_CHANNEL	(16)

// HSIM Integration
class Bridge : public CPU {
public:
    struct Queue {
        list<tlm_generic_payload *> q;

        unsigned int size() { return q.size(); }
    };

    using CPU::CPU;

    Bridge(const char *config_path, const char *path):CPU(config_path, path,
                     bind(&Bridge::ReadCallBack, this, placeholders::_1),
                     bind(&Bridge::WriteCallBack, this, placeholders::_1)) {};

    void ClockTick() override;

    void ReadCallBack(uint64_t addr);

    void WriteCallBack(uint64_t addr);

    // These functions are used to transfer tlm data.
    void sendCommand(tlm_generic_payload &trans);

    tlm_generic_payload *getCompletedCommand(void);

    void print_stats();
    void set_stats_prefix(const std::string& prefix);
    int getChannel(uint64_t address) const;
    sc_time getLastPimComputeStartTime() const { return last_compute_start_time; }
    sc_time getLastPimComputeDoneTime() const { return last_compute_done_time; }

private:
    Queue m_incomingQueue[MAX_CHANNEL];
    Queue m_outgoingQueue;

    vector<pair<uint64_t, tlm_generic_payload *>> m_readDone;
    vector<pair<uint64_t, tlm_generic_payload *>> m_writeDone;
    deque<pair<sc_time, tlm_generic_payload*>> m_pimComputeQueue;
    sc_time last_compute_start_time = sc_time(0, SC_NS);
    sc_time last_compute_done_time = sc_time(0, SC_NS);
};

}

#endif //PACKAGE_HW_BRIDGE_H
