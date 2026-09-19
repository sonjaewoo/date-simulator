#ifndef __CYCLE_H
#define __CYCLE_H

#include <vector>
#include <systemc>
#include <utilities/configurations.h>

extern Configurations cfgs;

struct NPUCycle {
    int64_t start = -1;
    int64_t end = -1;
    int64_t compute_cycle = -1;
};

class Logger {
public:
    void update_start(const std::string& layer, DeviceType type);
    void update_end(const std::string& layer, DeviceType type);
    // Explicit-cycle variants: the caller knows the exact event time, which may
    // differ from sc_time_stamp() when a request is queued before it runs.
    void update_start_at(const std::string& layer, DeviceType type, int64_t cycle);
    void update_end_at(const std::string& layer, DeviceType type, int64_t cycle);
    void update_npu_start(const std::string& layer, int64_t start);
    void update_npu_end(const std::string& layer, int64_t end, int64_t compute_cycle);
    void print_and_save_log(const std::string& filename = "");

private:
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> host_info;
    std::unordered_map<std::string, NPUCycle> npu_info;
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> dram_info;    
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> pim_info;

    void print_header(std::ostream& out, const std::string& title, const std::vector<std::string>& columns);
    void print_cycle_data(std::ostream& out, const std::string& layer, int64_t start, int64_t end, int64_t extra = -1);
};

#endif
