#include "performance_metrics.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {

std::string indent(unsigned int count)
{
    return std::string(count, ' ');
}

} // namespace

int64_t PerformanceMetrics::to_ns(sc_core::sc_time time)
{
    return static_cast<int64_t>(std::llround(time.to_seconds() * 1.0e9));
}

void PerformanceMetrics::configure(Device device, uint32_t channels,
                                   const std::string& channel_mode)
{
    auto& stats = device_stats(device);
    stats.channels = std::max(1U, channels);
    stats.channel_mode = channel_mode;
    stats.per_channel.resize(stats.channels);
}

void PerformanceMetrics::request_arrived(Device device, tlm::tlm_generic_payload* trans,
                                         uint64_t bytes, bool pim_compute)
{
    if (trans == nullptr) return;
    requests_.try_emplace(trans, Request{device, bytes, pim_compute, to_ns(sc_core::sc_time_stamp())});
}

void PerformanceMetrics::request_started(Device device, tlm::tlm_generic_payload* trans,
                                         sc_core::sc_time start, int channel)
{
    const auto request_it = requests_.find(trans);
    if (request_it == requests_.end()) return;

    auto& request = request_it->second;
    if (request.device != device || request.start_ns >= 0) return;

    request.start_ns = std::max(request.arrival_ns, to_ns(start));
    request.channel = channel;
    auto& traffic = request.pim_compute ? device_stats(device).compute : device_stats(device).memory;
    const uint64_t queue_wait = static_cast<uint64_t>(request.start_ns - request.arrival_ns);
    traffic.queue_wait_ns += queue_wait;
    traffic.max_queue_wait_ns = std::max(traffic.max_queue_wait_ns, queue_wait);
    traffic.queue_wait_samples.push_back(queue_wait);
}

void PerformanceMetrics::request_completed(Device device, tlm::tlm_generic_payload* trans,
                                           sc_core::sc_time end)
{
    const auto request_it = requests_.find(trans);
    if (request_it == requests_.end()) return;

    const Request request = request_it->second;
    requests_.erase(request_it);
    if (request.device != device || request.start_ns < 0) return;

    const int64_t end_ns = std::max(request.start_ns, to_ns(end));
    const Interval interval{request.start_ns, end_ns};
    auto& stats = device_stats(device);
    auto& traffic = request.pim_compute ? stats.compute : stats.memory;
    ++traffic.requests;
    traffic.bytes += request.bytes;
    traffic.service_intervals.push_back(interval);
    add_channel_service(stats, request, interval);
    if (request.pim_compute) {
        pim_compute_intervals_.push_back(interval);
        if (trans->get_layer().rfind("draft_pim.", 0) == 0) {
            pim_draft_compute_intervals_.push_back(interval);
        }
    }
}

void PerformanceMetrics::record_pim_resource_stall(sc_core::sc_time start,
                                                    sc_core::sc_time end,
                                                    bool draft_compute)
{
    const int64_t start_ns = to_ns(start);
    const int64_t end_ns = to_ns(end);
    if (end_ns <= start_ns) return;
    const uint64_t duration = static_cast<uint64_t>(end_ns - start_ns);
    if (draft_compute) pim_draft_compute_resource_stall_ns_ += duration;
    else pim_target_memory_resource_stall_ns_ += duration;
}

void PerformanceMetrics::record_npu_interval(sc_core::sc_time start, sc_core::sc_time end)
{
    const int64_t start_ns = to_ns(start);
    const int64_t end_ns = to_ns(end);
    if (end_ns > start_ns) npu_intervals_.push_back({start_ns, end_ns});
}

void PerformanceMetrics::record_npu_compute_interval(sc_core::sc_time start,
                                                      sc_core::sc_time end)
{
    const int64_t start_ns = to_ns(start);
    const int64_t end_ns = to_ns(end);
    if (end_ns > start_ns) npu_compute_intervals_.push_back({start_ns, end_ns});
}

void PerformanceMetrics::begin_parallel_episode(const std::string& episode_id,
                                                 sc_core::sc_time launch)
{
    if (episode_id.empty() || parallel_episode_index_.contains(episode_id)) return;
    parallel_episode_index_.emplace(episode_id, parallel_episodes_.size());
    parallel_episodes_.push_back({episode_id, to_ns(launch)});
}

void PerformanceMetrics::complete_parallel_npu_episode(const std::string& episode_id,
                                                        sc_core::sc_time end)
{
    const auto it = parallel_episode_index_.find(episode_id);
    if (it == parallel_episode_index_.end()) return;
    auto& episode = parallel_episodes_[it->second];
    episode.npu_end_ns = std::max(episode.launch_ns, to_ns(end));
}

void PerformanceMetrics::complete_parallel_pim_episode(const std::string& episode_id,
                                                        sc_core::sc_time end)
{
    const auto it = parallel_episode_index_.find(episode_id);
    if (it == parallel_episode_index_.end()) return;
    auto& episode = parallel_episodes_[it->second];
    episode.pim_end_ns = std::max(episode.launch_ns, to_ns(end));
}

uint64_t PerformanceMetrics::interval_union_ns(std::vector<Interval> intervals)
{
    intervals.erase(std::remove_if(intervals.begin(), intervals.end(), [](const Interval& value) {
        return value.end_ns <= value.start_ns;
    }), intervals.end());
    if (intervals.empty()) return 0;
    std::sort(intervals.begin(), intervals.end(), [](const Interval& lhs, const Interval& rhs) {
        return lhs.start_ns == rhs.start_ns ? lhs.end_ns < rhs.end_ns : lhs.start_ns < rhs.start_ns;
    });

    uint64_t total = 0;
    int64_t active_start = intervals.front().start_ns;
    int64_t active_end = intervals.front().end_ns;
    for (const auto& interval : intervals) {
        if (interval.start_ns > active_end) {
            total += static_cast<uint64_t>(active_end - active_start);
            active_start = interval.start_ns;
            active_end = interval.end_ns;
        } else {
            active_end = std::max(active_end, interval.end_ns);
        }
    }
    return total + static_cast<uint64_t>(active_end - active_start);
}

uint64_t PerformanceMetrics::interval_intersection_ns(std::vector<Interval> lhs,
                                                       std::vector<Interval> rhs)
{
    const auto normalize = [](std::vector<Interval>& intervals) {
        std::sort(intervals.begin(), intervals.end(), [](const Interval& a, const Interval& b) {
            return a.start_ns == b.start_ns ? a.end_ns < b.end_ns : a.start_ns < b.start_ns;
        });
        std::vector<Interval> merged;
        for (const auto& interval : intervals) {
            if (interval.end_ns <= interval.start_ns) continue;
            if (merged.empty() || interval.start_ns > merged.back().end_ns) {
                merged.push_back(interval);
            } else {
                merged.back().end_ns = std::max(merged.back().end_ns, interval.end_ns);
            }
        }
        intervals = std::move(merged);
    };
    normalize(lhs);
    normalize(rhs);

    size_t left = 0;
    size_t right = 0;
    uint64_t total = 0;
    while (left < lhs.size() && right < rhs.size()) {
        const int64_t begin = std::max(lhs[left].start_ns, rhs[right].start_ns);
        const int64_t end = std::min(lhs[left].end_ns, rhs[right].end_ns);
        if (end > begin) total += static_cast<uint64_t>(end - begin);
        if (lhs[left].end_ns < rhs[right].end_ns) ++left;
        else ++right;
    }
    return total;
}

uint64_t PerformanceMetrics::percentile95(std::vector<uint64_t> samples)
{
    if (samples.empty()) return 0;
    std::sort(samples.begin(), samples.end());
    const size_t index = static_cast<size_t>(std::ceil(samples.size() * 0.95)) - 1U;
    return samples[std::min(index, samples.size() - 1U)];
}

double PerformanceMetrics::percent(uint64_t numerator, uint64_t denominator)
{
    return denominator == 0 ? 0.0 : 100.0 * static_cast<double>(numerator)
        / static_cast<double>(denominator);
}

void PerformanceMetrics::write_traffic_json(std::ostream& out, const TrafficStats& stats,
                                            uint64_t simulation_ns, unsigned int spaces)
{
    const uint64_t busy_ns = interval_union_ns(stats.service_intervals);
    const uint64_t queue_count = stats.queue_wait_samples.size();
    out << indent(spaces) << "\"requests\": " << stats.requests << ",\n";
    out << indent(spaces) << "\"bytes\": " << stats.bytes << ",\n";
    out << indent(spaces) << "\"achieved_bandwidth_GBps\": "
        << (simulation_ns == 0 ? 0.0 : static_cast<double>(stats.bytes) / simulation_ns) << ",\n";
    out << indent(spaces) << "\"busy_ns\": " << busy_ns << ",\n";
    out << indent(spaces) << "\"busy_utilization_pct\": "
        << percent(busy_ns, simulation_ns) << ",\n";
    out << indent(spaces) << "\"queue_wait_ns\": " << stats.queue_wait_ns << ",\n";
    out << indent(spaces) << "\"average_queue_wait_ns\": "
        << (queue_count == 0 ? 0.0 : static_cast<double>(stats.queue_wait_ns) / queue_count) << ",\n";
    out << indent(spaces) << "\"p95_queue_wait_ns\": "
        << percentile95(stats.queue_wait_samples) << ",\n";
    out << indent(spaces) << "\"max_queue_wait_ns\": " << stats.max_queue_wait_ns << "\n";
}

void PerformanceMetrics::write_device_json(std::ostream& out, const DeviceStats& stats,
                                           uint64_t simulation_ns, unsigned int spaces)
{
    const uint64_t device_busy_ns = interval_union_ns([&]() {
        std::vector<Interval> intervals = stats.memory.service_intervals;
        intervals.insert(intervals.end(), stats.compute.service_intervals.begin(),
                         stats.compute.service_intervals.end());
        return intervals;
    }());
    out << indent(spaces) << "\"channels\": " << stats.channels << ",\n";
    out << indent(spaces) << "\"channel_accounting\": \"" << stats.channel_mode << "\",\n";
    out << indent(spaces) << "\"device_busy_ns\": " << device_busy_ns << ",\n";
    out << indent(spaces) << "\"device_busy_utilization_pct\": "
        << percent(device_busy_ns, simulation_ns) << ",\n";
    out << indent(spaces) << "\"memory\": {\n";
    write_traffic_json(out, stats.memory, simulation_ns, spaces + 2);
    out << indent(spaces) << "},\n";
    out << indent(spaces) << "\"compute\": {\n";
    write_traffic_json(out, stats.compute, simulation_ns, spaces + 2);
    out << indent(spaces) << "},\n";
    out << indent(spaces) << "\"per_channel\": [\n";
    for (size_t channel = 0; channel < stats.per_channel.size(); ++channel) {
        const auto& value = stats.per_channel[channel];
        const uint64_t busy_ns = interval_union_ns(value.service_intervals);
        out << indent(spaces + 2) << "{\"id\": " << channel
            << ", \"requests\": " << value.requests
            << ", \"bytes\": " << value.bytes
            << ", \"busy_ns\": " << busy_ns
            << ", \"busy_utilization_pct\": " << percent(busy_ns, simulation_ns)
            << "}" << (channel + 1U == stats.per_channel.size() ? "\n" : ",\n");
    }
    out << indent(spaces) << "]\n";
}

void PerformanceMetrics::write_json(const std::string& filename,
                                    sc_core::sc_time simulation_end) const
{
    if (filename.empty()) return;
    std::ofstream out(filename);
    if (!out.is_open()) {
        std::cerr << "[PerformanceMetrics] Cannot write " << filename << "\n";
        return;
    }

    const uint64_t simulation_ns = std::max<int64_t>(0, to_ns(simulation_end));
    const uint64_t npu_active_ns = interval_union_ns(npu_intervals_);
    const uint64_t npu_target_busy_ns = interval_union_ns(npu_compute_intervals_);
    const uint64_t pim_active_ns = interval_union_ns(pim_compute_intervals_);
    const uint64_t pim_draft_busy_ns = interval_union_ns(pim_draft_compute_intervals_);
    const uint64_t overlap_ns = interval_intersection_ns(npu_intervals_, pim_compute_intervals_);
    const uint64_t union_ns = npu_active_ns + pim_active_ns - overlap_ns;
    const uint64_t workload_busy_ns = pim_draft_busy_ns + npu_target_busy_ns;
    const double r_pim = workload_busy_ns == 0 ? 0.0
        : static_cast<double>(pim_draft_busy_ns) / static_cast<double>(workload_busy_ns);

    out << std::fixed << std::setprecision(3);
    out << "{\n";
    out << "  \"schema_version\": 3,\n";
    out << "  \"simulation_ns\": " << simulation_ns << ",\n";
    out << "  \"dram\": {\n";
    write_device_json(out, dram_, simulation_ns, 4);
    out << "  },\n";
    out << "  \"pim\": {\n";
    write_device_json(out, pim_, simulation_ns, 4);
    out << "  },\n";
    out << "  \"npu_pim_overlap\": {\n";
    out << "    \"npu_active_ns\": " << npu_active_ns << ",\n";
    out << "    \"pim_compute_active_ns\": " << pim_active_ns << ",\n";
    out << "    \"overlap_ns\": " << overlap_ns << ",\n";
    out << "    \"npu_covered_by_pim_pct\": " << percent(overlap_ns, npu_active_ns) << ",\n";
    out << "    \"pim_hidden_by_npu_pct\": " << percent(overlap_ns, pim_active_ns) << ",\n";
    out << "    \"intersection_over_union_pct\": " << percent(overlap_ns, union_ns) << "\n";
    out << "  },\n";
    out << "  \"workload_balance\": {\n";
    out << "    \"definition\": \"r_pim=pim_draft_busy_ns/(pim_draft_busy_ns+npu_target_busy_ns)\",\n";
    out << "    \"scope\": \"busy execution only: PIM draft compute service and NPU target compute; excludes queueing, memory-response waits, and dependency stalls\",\n";
    out << "    \"comparison_rule\": \"compare r_pim across workloads at a fixed reference partition, normally PIM:DRAM=2:2\",\n";
    out << "    \"pim_draft_busy_ns\": " << pim_draft_busy_ns << ",\n";
    out << "    \"npu_target_busy_ns\": " << npu_target_busy_ns << ",\n";
    out << "    \"total_busy_ns\": " << workload_busy_ns << ",\n";
    out << "    \"r_pim\": " << r_pim << ",\n";
    out << "    \"r_npu\": " << (1.0 - r_pim) << "\n";
    out << "  },\n";
    uint64_t eei_pim_only_ns = 0;
    uint64_t eei_npu_only_ns = 0;
    uint64_t eei_both_ns = 0;
    size_t completed_episodes = 0;
    for (const auto& episode : parallel_episodes_) {
        if (episode.npu_end_ns < episode.launch_ns || episode.pim_end_ns < episode.launch_ns) continue;
        const int64_t both_end = std::min(episode.npu_end_ns, episode.pim_end_ns);
        eei_both_ns += static_cast<uint64_t>(both_end - episode.launch_ns);
        eei_pim_only_ns += static_cast<uint64_t>(std::max<int64_t>(0, episode.pim_end_ns - both_end));
        eei_npu_only_ns += static_cast<uint64_t>(std::max<int64_t>(0, episode.npu_end_ns - both_end));
        ++completed_episodes;
    }
    const uint64_t eei_total_ns = eei_pim_only_ns + eei_npu_only_ns + eei_both_ns;
    const double aggregate_eei = eei_total_ns == 0 ? 0.0
        : static_cast<double>(static_cast<int64_t>(eei_pim_only_ns)
                              - static_cast<int64_t>(eei_npu_only_ns))
            / static_cast<double>(eei_total_ns);
    out << "  \"parallel_execution_imbalance\": {\n";
    out << "    \"definition\": \"EEI=(pim_only_ns-npu_only_ns)/(pim_only_ns+npu_only_ns+both_ns)\",\n";
    out << "    \"scope\": \"common stage launch to branch completion; includes queueing and memory waits on each branch\",\n";
    out << "    \"episodes_recorded\": " << parallel_episodes_.size() << ",\n";
    out << "    \"episodes_completed\": " << completed_episodes << ",\n";
    out << "    \"aggregate\": {\n";
    out << "      \"pim_only_ns\": " << eei_pim_only_ns << ",\n";
    out << "      \"npu_only_ns\": " << eei_npu_only_ns << ",\n";
    out << "      \"both_ns\": " << eei_both_ns << ",\n";
    out << "      \"total_parallel_ns\": " << eei_total_ns << ",\n";
    out << "      \"eei\": " << aggregate_eei << "\n";
    out << "    },\n";
    out << "    \"episodes\": [\n";
    bool first_episode = true;
    for (const auto& episode : parallel_episodes_) {
        if (episode.npu_end_ns < episode.launch_ns || episode.pim_end_ns < episode.launch_ns) continue;
        const int64_t both_end = std::min(episode.npu_end_ns, episode.pim_end_ns);
        const uint64_t both_ns = static_cast<uint64_t>(both_end - episode.launch_ns);
        const uint64_t pim_only_ns = static_cast<uint64_t>(std::max<int64_t>(0, episode.pim_end_ns - both_end));
        const uint64_t npu_only_ns = static_cast<uint64_t>(std::max<int64_t>(0, episode.npu_end_ns - both_end));
        const uint64_t total_ns = pim_only_ns + npu_only_ns + both_ns;
        const double eei = total_ns == 0 ? 0.0
            : static_cast<double>(static_cast<int64_t>(pim_only_ns) - static_cast<int64_t>(npu_only_ns))
                / static_cast<double>(total_ns);
        if (!first_episode) out << ",\n";
        first_episode = false;
        out << "      {\"episode\": \"" << episode.id
            << "\", \"launch_ns\": " << episode.launch_ns
            << ", \"npu_end_ns\": " << episode.npu_end_ns
            << ", \"pim_end_ns\": " << episode.pim_end_ns
            << ", \"pim_only_ns\": " << pim_only_ns
            << ", \"npu_only_ns\": " << npu_only_ns
            << ", \"both_ns\": " << both_ns
            << ", \"eei\": " << eei << "}";
    }
    out << "\n    ]\n";
    out << "  },\n";
    out << "  \"pim_side_stall\": {\n";
    out << "    \"draft_compute_resource_wait_ns\": " << pim_draft_compute_resource_stall_ns_ << ",\n";
    out << "    \"target_memory_resource_wait_ns\": " << pim_target_memory_resource_stall_ns_ << ",\n";
    out << "    \"total_resource_wait_ns\": "
        << (pim_draft_compute_resource_stall_ns_ + pim_target_memory_resource_stall_ns_) << "\n";
    out << "  }\n";
    out << "}\n";
    std::cout << "Performance breakdown saved to file: " << filename << "\n";
}

PerformanceMetrics::DeviceStats& PerformanceMetrics::device_stats(Device device)
{
    return device == Device::DRAM ? dram_ : pim_;
}

const PerformanceMetrics::DeviceStats& PerformanceMetrics::device_stats(Device device) const
{
    return device == Device::DRAM ? dram_ : pim_;
}

void PerformanceMetrics::add_channel_service(DeviceStats& stats, const Request& request,
                                             const Interval& interval)
{
    if (stats.per_channel.empty()) stats.per_channel.resize(std::max(1U, stats.channels));
    if (request.channel >= 0 && static_cast<size_t>(request.channel) < stats.per_channel.size()) {
        auto& channel = stats.per_channel[request.channel];
        ++channel.requests;
        channel.bytes += request.bytes;
        channel.service_intervals.push_back(interval);
        return;
    }

    // A fast tensor transaction is calibrated for the aggregate channel set.
    // Attribute its occupancy to every logical channel and split transferred
    // bytes, which makes the per-channel report an explicit balanced estimate.
    const uint64_t base_bytes = request.bytes / stats.per_channel.size();
    const uint64_t remainder = request.bytes % stats.per_channel.size();
    for (size_t channel_id = 0; channel_id < stats.per_channel.size(); ++channel_id) {
        auto& channel = stats.per_channel[channel_id];
        ++channel.requests;
        channel.bytes += base_bytes + (channel_id < remainder ? 1U : 0U);
        channel.service_intervals.push_back(interval);
    }
}
