#ifndef __TRACE_GENERATOR_H
#define __TRACE_GENERATOR_H

#include <systemc.h>
#include <unordered_set>
#include <utilities/common.h>
#include <utilities/configurations.h>

extern Configurations cfgs;

enum class TraceType {
    WRITE,
    READ,
    COMPUTE,
    PIM,
    SPECULATIVE,
    TERMINATE
};

class TraceGenerator {
public:   
    struct Trace {
        virtual ~Trace() = default;
        TraceType type;
        std::string layer;
        Trace(TraceType t, std::string l) : type(t), layer(l) {}
    };

    struct MemoryTrace : public Trace {
        unsigned int id, src, dst, size;
        uint64_t address;
        MemoryTrace(TraceType t, unsigned int i, unsigned int s, unsigned int d, unsigned int z, uint64_t a, std::string l)
        : Trace(t, l), id(i), src(s), dst(d), size(z), address(a) {}  
    };

    struct ComputeTrace : public Trace {
        unsigned int device;
        ComputeTrace(TraceType t, unsigned int d, std::string l)
        : Trace(t, l), device(d) {}  
    };

    struct SimTrace : public Trace {
        SimTrace(TraceType t, std::string l)
        : Trace(t, l) {}  
    };

    struct SpeculativeTrace : public Trace {
        SpeculativeTrace(std::string scenario) : Trace(TraceType::SPECULATIVE, std::move(scenario)) {}
    };

    struct SpeculativeControl {
        uint32_t sample_index;
        uint32_t iteration;
        uint32_t prefix_len_before;
        uint32_t prefix_len_after;
        uint32_t gamma;
        uint32_t target_query_tokens;
        uint32_t target_kv_cache_tokens;
        uint32_t target_kv_write_tokens;
        // PEARL v2 traces distinguish the number of decoder tokens executed
        // by the draft actor from its requested generation horizon.  For a
        // warm cache this is gamma, while a first/prefill iteration may have
        // to consume more than one uncached input token first.
        uint32_t draft_query_tokens_per_proposal;
        uint32_t draft_proposal_count;
        uint32_t draft_kv_cache_tokens;
        uint32_t draft_kv_write_tokens;
        uint32_t target_input_tokens;
        uint32_t draft_input_tokens;
        uint32_t target_kv_cache_tokens_after_forward;
        uint32_t draft_kv_cache_tokens_after_forward;
        uint32_t target_kv_cache_tokens_after_decision;
        uint32_t draft_kv_cache_tokens_after_decision;
        uint32_t target_generated_tokens;
        uint32_t draft_generated_tokens;
        uint32_t draft_forward_steps;
        uint32_t trace_version;
        bool prefill_excluded;
        std::string mode_before;
        std::string mode_after;
        std::string decision;
        uint32_t accepted_tokens;
        int64_t rollback_to;
        // PEARL uses a same-round proposal/verification record.  SSD uses
        // the same record shape, but its draft work is a precompute for the
        // following target verification round.
        std::string algorithm;
        int64_t target_depends_on_draft_iteration;
        int64_t draft_precompute_for_iteration;
        bool draft_overlaps_target;
        // SSD asynchronous drafting runs a K-step tree decode.  At every
        // tree step, fan_out_list describes how many branches originate at
        // each of the K+1 verification positions.  PIM may serialize those
        // branches, but their attention context must retain this topology.
        uint32_t draft_tree_steps;
        uint32_t draft_tree_query_tokens;
        std::vector<uint32_t> draft_fan_out_list;
    };

    struct PimTrace : public Trace {
        std::string cmd;
        unsigned int next;
        PimTrace(TraceType t, std::string l, std::string c, unsigned int n)
        : Trace(t, l), cmd(c), next(n) {}  
    };

    TraceGenerator();
    std::deque<std::shared_ptr<Trace>> trace_queue;
    std::unordered_set<std::string> layers;

    std::deque<std::shared_ptr<TraceGenerator::Trace>>& generate_trace();
    const std::vector<SpeculativeControl>& get_speculative_control() const { return speculative_control; }
    
    void add_trace(TraceType type, const std::string& layer, unsigned int device);
    void add_gemv_trace(const std::string& layer, const std::vector<std::string>& cmds, unsigned int next);
    void add_rope_trace(const std::string& layer, const std::vector<std::string>& cmds);
    void add_memory_trace(TraceType type, const std::string& layer, DeviceType src, const std::vector<DeviceType>& dst_list);
    void add_host_trace(const std::string& layer, const std::vector<DeviceType>& r_targets, const std::vector<DeviceType>& w_targets);
    void add_speculative_trace(const std::string& scenario);

private:
    SyncMap host_sync_map;
    WaitMap host_wait_map;
    std::vector<SpeculativeControl> speculative_control;

    void load_speculative_control_trace();
};

#endif
