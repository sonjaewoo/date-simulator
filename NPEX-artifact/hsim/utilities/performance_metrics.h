#ifndef PERFORMANCE_METRICS_H
#define PERFORMANCE_METRICS_H

#include <cstdint>
#include <iosfwd>
#include <string>
#include <unordered_map>
#include <vector>

#include <systemc>
#include <tlm.h>

// Runtime breakdown for the heterogeneous memory system.  A request's queue
// time is measured from arrival at an HSIM wrapper to the time the modeled
// resource can begin service.  Consequently it includes wrapper/resource
// contention, but not time before the request reaches the wrapper.
class PerformanceMetrics {
public:
    enum class Device { DRAM, PIM };

    // kAllChannels is used by tensor-sized fast-mode transactions and PIM
    // compute commands whose latency model represents all configured channels.
    static constexpr int kAllChannels = -1;

    void configure(Device device, uint32_t channels, const std::string& channel_mode);

    void request_arrived(Device device, tlm::tlm_generic_payload* trans,
                         uint64_t bytes, bool pim_compute = false);
    void request_started(Device device, tlm::tlm_generic_payload* trans,
                         sc_core::sc_time start, int channel = kAllChannels);
    void request_completed(Device device, tlm::tlm_generic_payload* trans,
                           sc_core::sc_time end);

    // A unified-memory resource conflict is a subset of normal queue wait;
    // retain it separately so the PIM-side source of a stall is explicit.
    void record_pim_resource_stall(sc_core::sc_time start, sc_core::sc_time end,
                                   bool draft_compute);

    void record_npu_interval(sc_core::sc_time start, sc_core::sc_time end);
    // Records only target-NPU compute execution.  Memory-response waits are
    // intentionally excluded so workload balance is not a critical-path or
    // stall metric.
    void record_npu_compute_interval(sc_core::sc_time start, sc_core::sc_time end);

    // A speculative-decoding stage launches one target-NPU branch and one
    // draft-PIM branch, then waits for both.  These hooks capture their
    // common launch and individual completion points so exposed execution can
    // be measured on the stage critical path (including resource queueing).
    void begin_parallel_episode(const std::string& episode_id, sc_core::sc_time launch);
    void complete_parallel_npu_episode(const std::string& episode_id, sc_core::sc_time end);
    void complete_parallel_pim_episode(const std::string& episode_id, sc_core::sc_time end);

    // Writes a machine-readable report.  All times are simulation nanoseconds
    // and bandwidth is decimal GB/s (bytes/ns).
    void write_json(const std::string& filename, sc_core::sc_time simulation_end) const;

private:
    struct Interval {
        int64_t start_ns{0};
        int64_t end_ns{0};
    };

    struct ChannelStats {
        uint64_t bytes{0};
        uint64_t requests{0};
        std::vector<Interval> service_intervals;
    };

    struct Request {
        Device device{Device::DRAM};
        uint64_t bytes{0};
        bool pim_compute{false};
        int64_t arrival_ns{0};
        int64_t start_ns{-1};
        int channel{kAllChannels};
    };

    struct TrafficStats {
        uint64_t requests{0};
        uint64_t bytes{0};
        uint64_t queue_wait_ns{0};
        uint64_t max_queue_wait_ns{0};
        std::vector<uint64_t> queue_wait_samples;
        std::vector<Interval> service_intervals;
    };

    struct DeviceStats {
        uint32_t channels{1};
        std::string channel_mode{"physical_address_mapped"};
        TrafficStats memory;
        TrafficStats compute;
        std::vector<ChannelStats> per_channel;
    };

    struct ParallelEpisode {
        std::string id;
        int64_t launch_ns{0};
        int64_t npu_end_ns{-1};
        int64_t pim_end_ns{-1};
    };

    static int64_t to_ns(sc_core::sc_time time);
    static uint64_t interval_union_ns(std::vector<Interval> intervals);
    static uint64_t interval_intersection_ns(std::vector<Interval> lhs,
                                             std::vector<Interval> rhs);
    static uint64_t percentile95(std::vector<uint64_t> samples);
    static double percent(uint64_t numerator, uint64_t denominator);
    static void write_traffic_json(std::ostream& out, const TrafficStats& stats,
                                   uint64_t simulation_ns, unsigned int indent);
    static void write_device_json(std::ostream& out, const DeviceStats& stats,
                                  uint64_t simulation_ns, unsigned int indent);

    DeviceStats& device_stats(Device device);
    const DeviceStats& device_stats(Device device) const;
    void add_channel_service(DeviceStats& stats, const Request& request,
                             const Interval& interval);

    DeviceStats dram_;
    DeviceStats pim_;
    std::unordered_map<tlm::tlm_generic_payload*, Request> requests_;
    std::vector<Interval> npu_intervals_;
    std::vector<Interval> npu_compute_intervals_;
    std::vector<Interval> pim_compute_intervals_;
    std::vector<Interval> pim_draft_compute_intervals_;
    std::vector<ParallelEpisode> parallel_episodes_;
    std::unordered_map<std::string, size_t> parallel_episode_index_;
    uint64_t pim_draft_compute_resource_stall_ns_{0};
    uint64_t pim_target_memory_resource_stall_ns_{0};
};

#endif
