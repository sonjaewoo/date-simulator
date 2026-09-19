#ifndef CONFIGURATIONS_H
#define CONFIGURATIONS_H

#include <map>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <filesystem>
#include <json.h>
#include <utilities/common.h>

typedef enum _LOG_LEVEL {
    LOG_DEBUG = 0,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR,
    LOG_OFF
} LOG_LEVEL;

struct MemInfo {
    uint64_t shared_offset;
    uint64_t input_offset;
    uint64_t output_offset;
    uint64_t wb_offset;
    uint64_t buf_offset;
};

struct NetworkInfo {
    std::string name;
    MemInfo meminfo;
    std::string out_path;
    std::string prefix;
    bool precompiled;
};

struct CompileInfo {
    bool quantized;
    std::string midap_compiler;
    unsigned packet_size;
    unsigned midap_level;
    uint32_t fmem_bank_num;
    uint32_t fmem_bank_size;
    uint32_t cim_num;
    uint32_t wmem_size;
    uint32_t ewmem_size;
    NetworkInfo net_info;
    std::string additional_flags;
};

struct DRAMInfo {
    uint32_t channels;
    uint32_t capacity;
    std::string backend;
    std::string type;
    std::string config;
    std::string preset;
    double freq;
};

struct PIMInfo {
    uint32_t channels;
    uint32_t capacity;
    std::string backend;
    std::string type;
    std::string config;
    std::string preset;
    double freq;
};

struct SyntheticNPUInfo {
    bool enabled{false};
    uint32_t compute_ns_per_layer{0};
    uint32_t weight_read_bytes_per_layer{0};
    uint32_t output_write_bytes_per_layer{0};
    uint32_t max_inflight_requests{256};
    std::string model_profile;
    bool fast_layer_mode{false};
    // "macro" collapses every decoder layer into one transaction (fastest);
    // "layer" keeps the analytical timing model but emits one transaction and
    // one cycle-log record per decoder layer.
    std::string fast_layer_granularity{"macro"};
    // "analytic" preserves the legacy fixed-bandwidth delay.  The calibrated
    // mode measures a short stream in the selected DRAMsim3 configuration and
    // uses that timing for every tensor-sized macro transfer.
    std::string fast_memory_backend{"analytic"};
    // "native" uses the DRAMsim3 multi-channel address mapper.  "linear_logical"
    // calibrates one physical channel and assumes perfectly striped, independent
    // logical channels; it therefore supports non-power-of-two counts such as 3.
    std::string fast_memory_channel_model{"native"};
    // Utilization of every channel added after the first in linear_logical
    // mode.  1.0 is the ideal upper bound; lower values model diminishing
    // transfer-BW gains from imperfect striping and shared overheads.
    double fast_memory_channel_efficiency{1.0};
    uint32_t dram_macro_calibration_requests{4096};
    double fast_memory_bandwidth_GBps{25.6};
    uint32_t fast_memory_fixed_latency_ns{100};
    // Analytical NPU compute model.  A value of 1024 means 1024 MACs/cycle.
    bool analytical_compute_enabled{true};
    uint32_t macs_per_cycle{1024};
    double compute_frequency_GHz{0.5};
};

struct SyntheticPIMInfo {
    bool enabled{false};
    uint32_t layer_count{0};
    uint32_t compute_ns_per_layer{0};
    uint32_t commands_per_layer{1};
    std::string command{"mac"};
    std::string model_profile;
    bool aggregate_micro_commands{false};
    bool fast_layer_mode{false};
    std::string fast_layer_granularity{"macro"};
};

// One decoder-layer operation from configs/model/*.json.  Weight bytes are
// explicit so a trace does not need to infer tensor dimensions at runtime.
struct ModelOperationInfo {
    std::string name;
    uint64_t weight_read_bytes{0};
    uint32_t compute_share{1};
    // MAC count for one decoded query token.  Attention operations additionally
    // use macs_per_kv_token for every query-token/KV-token pair.
    uint64_t macs_per_token{0};
    uint64_t macs_per_kv_token{0};
};

struct ModelLayerProfile {
    std::string name;
    uint32_t layer_count{0};
    uint64_t output_write_bytes{0};
    // K and V together, for one cached token in one decoder layer.
    uint64_t kv_cache_bytes_per_token{0};
    // Context tokens represented by one PIM attention tile.
    uint32_t attention_context_tile_tokens{128};
    std::vector<ModelOperationInfo> operations;
};

struct SpeculativeInfo {
    uint32_t scenario_id{0};
    std::string control_trace_file;
    // PEARL appends multiple samples to one control JSONL and restarts the
    // iteration counter at zero for each sample. -1 replays all samples.
    int32_t sample_index{-1};
    uint32_t skip_warmup_iterations{0};
    uint32_t max_control_iterations{0}; // 0 means process the full trace
    bool exclude_prefill{false};
    std::string draft_profile_layer;
};

struct DelayCount {
    unsigned int delay;
    unsigned int count;
};

struct PIMCmdProfile {
    std::unordered_map<std::string, DelayCount> cmd_map;
    std::vector<std::string> ordered_cmd_name;
    // T(C) / T(1) from the operation's channel_scaling profile.
    double channel_latency_scale{1.0};
};

class Configurations {
public:
    Configurations();

    Json::Value parse_json(const std::string& file_name);

    void init_dram();
    void init_system();
    void init_compiler();
    void init_sync_wait_map();
    void init_configurations();
    void compile_network();
    void init_host_profile();
    void init_pim_profile(const std::string& pim_type);
    
    bool pim_enabled()  { return enable_pim; }
    bool dram_enabled() { return enable_dram; }
    bool synthetic_npu_enabled() { return synthetic_npu.enabled; }
    bool synthetic_pim_enabled() { return synthetic_pim.enabled; }
    bool fast_layer_mode_enabled() const {
        return synthetic_npu.fast_layer_mode || synthetic_pim.fast_layer_mode;
    }

    uint32_t get_scenario() { return scenario_id; }
    bool speculative_enabled() { return execution_mode == "speculative"; }
    std::string get_execution_mode() { return execution_mode; }
    uint32_t get_packet_size() { return compileinfo.packet_size; }
    uint32_t get_midap_level() { return compileinfo.midap_level; }
    uint32_t get_dram_req_size() { return dram_req_size; }
    uint64_t get_dram_capacity() { return draminfo.capacity; }
    uint32_t get_dram_channels() { return draminfo.channels; }
    uint32_t get_pim_channels() { return piminfo.channels; }

    MemInfo get_meminfo() { return compileinfo.net_info.meminfo; }
    NetworkInfo get_netinfo() { return compileinfo.net_info; }
    SyncMap get_host_sync_map() { return host_sync_map; }
    WaitMap get_host_wait_map() { return host_wait_map; }
    SyncMap get_pim_sync_map() { return pim_sync_map; }
    WaitMap get_pim_wait_map() { return pim_wait_map; }

    double get_dram_freq() { return draminfo.freq; }
    double get_pim_freq() { return piminfo.freq; }

    double get_host_freq() { return host_frequency; }

    std::string get_cycle_log_file() { return cycle_log_file; }
    std::string get_performance_stats_file() const {
        return performance_stats_file.empty()
            ? (cycle_log_file.empty() ? "performance_breakdown.json"
                                      : cycle_log_file + ".breakdown.json")
            : performance_stats_file;
    }

    std::string get_pim_config() { return piminfo.config; }
    std::string get_pim_type() { return piminfo.type; }
    std::string get_pim_preset() { return piminfo.preset; }

    std::string get_dram_config() { return draminfo.config; }
    std::string get_dram_type() { return draminfo.type; }
    std::string get_dram_preset() { return draminfo.preset; }
    std::string get_dram_backend() { return draminfo.backend; }

    std::string get_memory_structure() { return memory_structure; }

    std::string get_net_name() { return compileinfo.net_info.name; }
    std::string get_compile_dir() { return compileinfo.net_info.out_path; }
    std::string get_compile_prefix() { return compileinfo.net_info.prefix; }
    std::string get_target_model() { return target_model; }
    
    const std::unordered_map<unsigned int, bool>& get_pim_sync_id_map() { return pim_sync_id_map; }
    const std::unordered_map<unsigned int, bool>& get_pim_wait_id_map() { return pim_wait_id_map; }
    const std::unordered_map<std::string, unsigned int>& get_layer_tile_map() { return layer_tile_map; }
    const std::unordered_map<std::string, unsigned int>& get_host_profile() { return host_profile; }
    const std::unordered_map<std::string, PIMCmdProfile>& get_pim_profile() { return pim_profile; }
    const std::tuple<unsigned int, unsigned int>& get_mode_change_delay() { return mode_change_delay; }
    void load_sync_wait_info(const std::string& file_name, BaseMap& info_map);
    void print_configurations();

    unsigned int get_attn_head_num() { return num_attn_heads; }
    unsigned int get_decoder_block_count() { return decoder_block; }
    SyntheticNPUInfo get_synthetic_npu_info() { return synthetic_npu; }
    SyntheticPIMInfo get_synthetic_pim_info() { return synthetic_pim; }
    SpeculativeInfo get_speculative_info() { return speculative; }
    const ModelLayerProfile& get_target_layer_profile() const { return target_layer_profile; }
    const ModelLayerProfile& get_draft_layer_profile() const { return draft_layer_profile; }

private:
    bool enable_pim;
    bool enable_dram;    

    uint32_t scenario_id;
    std::string execution_mode;
    uint32_t dram_req_size;
    std::string memory_structure;
    std::string cycle_log_file;
    std::string performance_stats_file;
    CompileInfo compileinfo;
    DRAMInfo draminfo;
    PIMInfo piminfo;
    SyntheticNPUInfo synthetic_npu;
    SyntheticPIMInfo synthetic_pim;
    SpeculativeInfo speculative;
    ModelLayerProfile target_layer_profile;
    ModelLayerProfile draft_layer_profile;

    unsigned int dram_to_pim_delay;
    unsigned int pim_to_dram_delay;    
    unsigned int input_sequence_length;
    unsigned int output_sequence_length;
    unsigned int decoder_block;
    unsigned int num_attn_heads;
    double host_frequency;

    std::string target_model;

    std::map<std::string, int> dram_list;
    std::unordered_map<std::string, unsigned int> layer_tile_map;

    std::unordered_map<std::string, unsigned int> host_profile;
    std::unordered_map<std::string, PIMCmdProfile> pim_profile;
    std::unordered_map<unsigned int, bool> pim_sync_id_map;
    std::unordered_map<unsigned int, bool> pim_wait_id_map;
    std::tuple<unsigned int, unsigned int> mode_change_delay;    

    SyncMap host_sync_map;
    WaitMap host_wait_map;
    SyncMap pim_sync_map;
    WaitMap pim_wait_map;

    ModelLayerProfile load_model_layer_profile(const std::string& file_name);
};

#endif //CONFIGURATIONS_H
