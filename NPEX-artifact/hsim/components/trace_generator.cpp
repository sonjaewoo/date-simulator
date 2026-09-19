#include "trace_generator.h"

#include <array>

#include <scenarios/speculative/s1.h>

TraceGenerator::TraceGenerator() {
    host_sync_map = cfgs.get_host_sync_map();
    host_wait_map = cfgs.get_host_wait_map();
    if (cfgs.speculative_enabled()) {
        load_speculative_control_trace();
    }
};

std::deque<std::shared_ptr<TraceGenerator::Trace>>& TraceGenerator::generate_trace()
{
    switch (cfgs.get_speculative_info().scenario_id) {
        case 1: return generate_speculative_s1(*this);
        default: throw std::runtime_error("Invalid speculative scenario.");
    }
}

void TraceGenerator::add_memory_trace(TraceType type, const std::string& layer, DeviceType src, const std::vector<DeviceType>& dst_list)
{
    for (const auto& dst : dst_list) {
        BaseMap* base_map = nullptr;

        if (src == HOST) {
            base_map = (type == TraceType::READ) ? &host_wait_map : &host_sync_map;
        }

        bool found = false;

        for (auto it = base_map->begin(); it != base_map->end(); ++it) {
            if (layer == "RoPE_") {
                if (it->first.find(layer) != std::string::npos) {
                    found = true;
                    const auto& [id, size, address] = it->second;
                    trace_queue.push_back(std::make_shared<MemoryTrace>(type, id, src, dst, size, address, it->first));
                }
            }
            else if (it->first == layer) {
                found = true;
                const auto& [id, size, address] = it->second;
                trace_queue.push_back(std::make_shared<MemoryTrace>(type, id, src, dst, size, address, it->first));
            }
        }

        if (!found) {
            std::cerr << "[TraceGenerator][ERROR] No entries found for " << layer << " in map!" << std::endl;
        }
    }
}

void TraceGenerator::add_trace(TraceType type, const std::string& layer, unsigned int device) {
    if (type == TraceType::COMPUTE) {
        trace_queue.push_back(std::make_shared<ComputeTrace>(type, device, layer));
    } else if (type == TraceType::TERMINATE) {
        trace_queue.push_back(std::make_shared<SimTrace>(type, layer));
    }
}

void TraceGenerator::add_speculative_trace(const std::string& scenario)
{
    trace_queue.push_back(std::make_shared<SpeculativeTrace>(scenario));
}

void TraceGenerator::load_speculative_control_trace()
{
    const auto spec = cfgs.get_speculative_info();
    std::ifstream file(spec.control_trace_file);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open speculative control trace: " + spec.control_trace_file);
    }

    Json::CharReaderBuilder builder;
    std::string line;
    uint32_t line_no = 0;
    std::vector<std::vector<SpeculativeControl>> samples;
    std::vector<SpeculativeControl> current_sample;
    bool current_sample_has_explicit_id = false;
    uint64_t current_trace_sample_id = 0;
    while (std::getline(file, line)) {
        ++line_no;
        if (line.empty()) continue;
        Json::Value root;
        std::string errors;
        const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
        if (!reader->parse(line.data(), line.data() + line.size(), &root, &errors)) {
            throw std::runtime_error("Invalid JSON in speculative control trace line " + std::to_string(line_no));
        }
        // A future SSD task trace may share the JSONL directory with this
        // control trace.  Only control records define replay iterations.
        if (root.isMember("event") && root["event"].asString() != "control") continue;
        const uint32_t iteration = root["iteration"].asUInt();
        const std::string algorithm = root.get("algorithm", "pearl").asString();
        const uint32_t prefix_before = root["prefix_len_before"].asUInt();
        const uint32_t prefix_after = root["prefix_len_after"].asUInt();
        const uint32_t trace_version = root.get("trace_version", 1).asUInt();
        const bool prefill_excluded = root.get("prefill_excluded", false).asBool();
        if (trace_version >= 2) {
            const std::array<const char*, 14> required_fields = {
                "target_input_tokens", "target_kv_cache_tokens",
                "target_uncached_query_tokens", "target_kv_write_tokens",
                "draft_input_tokens", "draft_kv_cache_tokens",
                "draft_uncached_query_tokens", "draft_forward_steps",
                "draft_decoder_tokens", "draft_kv_write_tokens",
                "target_kv_cache_tokens_after_forward",
                "draft_kv_cache_tokens_after_forward",
                "target_kv_cache_tokens_after_decision",
                "draft_kv_cache_tokens_after_decision",
            };
            for (const char* field : required_fields) {
                if (!root.isMember(field)) {
                    throw std::runtime_error("PEARL v2 control trace line "
                        + std::to_string(line_no) + " is missing " + field);
                }
            }
            const uint32_t target_input = root["target_input_tokens"].asUInt();
            const uint32_t target_cache = root["target_kv_cache_tokens"].asUInt();
            const uint32_t target_uncached = root["target_uncached_query_tokens"].asUInt();
            const uint32_t draft_input = root["draft_input_tokens"].asUInt();
            const uint32_t draft_cache = root["draft_kv_cache_tokens"].asUInt();
            const uint32_t draft_uncached = root["draft_uncached_query_tokens"].asUInt();
            const uint32_t draft_steps = root["draft_forward_steps"].asUInt();
            const uint32_t draft_decoder = root["draft_decoder_tokens"].asUInt();
            if (target_cache > target_input || draft_cache > draft_input
                || target_uncached != target_input - target_cache
                || draft_uncached != draft_input - draft_cache
                || target_uncached == 0 || draft_steps == 0
                || draft_decoder != draft_uncached + draft_steps - 1) {
                throw std::runtime_error("Invalid PEARL v2 cache/work shape in control trace line "
                    + std::to_string(line_no));
            }
        }
        // SSD's draft starts after the target has launched and therefore sees
        // the post-verification prefix.  New SSD traces state this field
        // explicitly; the fallback also makes older SSD traces replay with
        // the correct steady-state context approximation.
        const uint32_t draft_kv_tokens = root.get(
            "draft_precompute_kv_cache_tokens",
            root.get("draft_kv_cache_tokens",
                     algorithm == "ssd" ? Json::Value(prefix_after) : Json::Value(prefix_before))).asUInt();
        const uint32_t draft_proposal_count = root.get(
            "draft_decoder_tokens", root.get("draft_proposal_count", root["gamma"])).asUInt();
        const uint32_t draft_tree_steps = root.get(
            "draft_tree_steps", algorithm == "ssd" ? root["gamma"] : Json::Value(1)).asUInt();
        const uint32_t draft_tree_query_tokens = root.get(
            "draft_tree_query_tokens", draft_proposal_count).asUInt();
        std::vector<uint32_t> draft_fan_out_list;
        if (root.isMember("draft_fan_out_list")) {
            const auto& fan_out = root["draft_fan_out_list"];
            if (!fan_out.isArray()) {
                throw std::runtime_error("SSD draft_fan_out_list must be an array in control trace line "
                    + std::to_string(line_no));
            }
            for (const auto& count : fan_out) {
                const uint32_t branch_count = count.asUInt();
                if (branch_count == 0) {
                    throw std::runtime_error("SSD draft_fan_out_list contains a zero branch count in control trace line "
                        + std::to_string(line_no));
                }
                draft_fan_out_list.push_back(branch_count);
            }
        }
        if (algorithm == "ssd") {
            if (draft_tree_steps == 0 || draft_tree_query_tokens == 0 || draft_fan_out_list.empty()) {
                throw std::runtime_error("SSD control trace must provide non-zero draft_tree_steps, "
                    "draft_tree_query_tokens, and draft_fan_out_list in line " + std::to_string(line_no));
            }
            uint64_t fan_out_total = 0;
            for (const uint32_t count : draft_fan_out_list) fan_out_total += count;
            if (fan_out_total != draft_tree_query_tokens
                || draft_proposal_count != draft_tree_query_tokens) {
                throw std::runtime_error("SSD tree shape does not match draft proposal count in control trace line "
                    + std::to_string(line_no));
            }
        }
        // A trace-aware SSD benchmark writes a stable request/sample id.
        // Legacy PEARL/SSD traces omit it and reset iteration to zero at each
        // sample boundary, so retain that convention as the fallback.
        const bool has_explicit_sample_id = root.isMember("sample_index");
        const uint64_t trace_sample_id = has_explicit_sample_id
            ? root["sample_index"].asUInt64() : 0;
        const bool starts_new_sample = !current_sample.empty()
            && ((has_explicit_sample_id && (!current_sample_has_explicit_id
                                             || trace_sample_id != current_trace_sample_id))
                || (!has_explicit_sample_id && iteration == 0));
        if (starts_new_sample) {
            samples.push_back(std::move(current_sample));
            current_sample.clear();
        }
        if (current_sample.empty()) {
            current_sample_has_explicit_id = has_explicit_sample_id;
            current_trace_sample_id = trace_sample_id;
        }
        SpeculativeControl control{
            0,
            iteration,
            prefix_before,
            prefix_after,
            root["gamma"].asUInt(),
            // v2 records the actual target cache miss.  In PEARL post-verify
            // this is gamma even though the Python model calls generate(...,
            // 1): the forward first consumes the gamma speculative tokens.
            root.get("target_uncached_query_tokens", root.get("target_query_tokens", 1)).asUInt(),
            root.get("target_kv_cache_tokens", prefix_before).asUInt(),
            root.get("target_kv_write_tokens",
                     root.get("target_uncached_query_tokens", root.get("target_query_tokens", 1))).asUInt(),
            root.get("draft_query_tokens_per_proposal", 1).asUInt(),
            // draft_decoder_tokens includes an initial multi-token cache
            // fill, if any.  v1 traces retain their proposal-count meaning.
            draft_proposal_count,
            draft_kv_tokens,
            root.get("draft_kv_write_tokens",
                     root.get("draft_decoder_tokens", root.get("draft_proposal_count", root["gamma"]))).asUInt(),
            root.get("target_input_tokens", prefix_before).asUInt(),
            root.get("draft_input_tokens", prefix_before).asUInt(),
            root.get("target_kv_cache_tokens_after_forward", 0).asUInt(),
            root.get("draft_kv_cache_tokens_after_forward", 0).asUInt(),
            root.get("target_kv_cache_tokens_after_decision", 0).asUInt(),
            root.get("draft_kv_cache_tokens_after_decision", 0).asUInt(),
            root.get("target_generated_tokens", 1).asUInt(),
            root.get("draft_generated_tokens", root["gamma"]).asUInt(),
            root.get("draft_forward_steps", root["gamma"]).asUInt(),
            trace_version,
            prefill_excluded,
            root["mode_before"].asString(),
            root["mode_after"].asString(),
            root["decision"].asString(),
            root["accepted_tokens"].asUInt(),
            root["rollback_to"].isNull() ? -1 : root["rollback_to"].asInt64(),
            algorithm,
            root.get("target_depends_on_draft_iteration", static_cast<int64_t>(iteration) - 1).asInt64(),
            root.get("draft_precompute_for_iteration", static_cast<int64_t>(iteration) + 1).asInt64(),
            root.get("draft_overlaps_target", algorithm == "ssd").asBool(),
            draft_tree_steps,
            draft_tree_query_tokens,
            std::move(draft_fan_out_list),
        };
        current_sample.push_back(std::move(control));
    }
    if (!current_sample.empty()) {
        samples.push_back(std::move(current_sample));
    }
    if (spec.exclude_prefill) {
        for (uint32_t sample_id = 0; sample_id < samples.size(); ++sample_id) {
            auto& sample = samples[sample_id];
            if (sample.empty()) continue;
            auto& first = sample.front();
            if (first.trace_version < 2) {
                throw std::runtime_error("speculative.exclude_prefill requires a PEARL v2 trace with actor cache fields");
            }
            // New extractors prefill P-1 tokens before starting the control
            // trace.  Legacy v2 traces instead include that prefill in their
            // first pre-verify forward (cache=0, query=P).  Normalize only
            // the latter and preserve the actual pre-verify work.
            if (first.prefill_excluded) continue;
            if (first.prefix_len_before == 0 || first.target_kv_cache_tokens != 0
                || first.draft_kv_cache_tokens != 0) {
                throw std::runtime_error("Cannot identify prompt prefill in the first PEARL v2 record; "
                    "set prefill_excluded=true in the trace if it was already removed");
            }
            const uint32_t warm_cache_tokens = first.prefix_len_before - 1;
            first.target_input_tokens = first.prefix_len_before;
            first.target_kv_cache_tokens = warm_cache_tokens;
            first.target_query_tokens = 1;
            first.target_kv_write_tokens = 1;
            first.target_kv_cache_tokens_after_forward = first.prefix_len_before;
            first.draft_input_tokens = first.prefix_len_before;
            first.draft_kv_cache_tokens = warm_cache_tokens;
            first.draft_query_tokens_per_proposal = 1;
            first.draft_proposal_count = first.draft_forward_steps;
            first.draft_kv_write_tokens = first.draft_forward_steps;
            first.draft_kv_cache_tokens_after_forward = warm_cache_tokens + first.draft_forward_steps;
            first.prefill_excluded = true;
            std::cout << "[Speculative] excluded prompt prefill for sample " << sample_id
                      << ": initial cache=" << warm_cache_tokens
                      << ", preserved first pre-verify target_q=1"
                      << ", draft_tokens=" << first.draft_proposal_count << "\n";
        }
    }
    // A PEARL v2 record reports the real per-actor cache state after
    // verification/rollback.  The following record must start from exactly
    // that state; otherwise the trace would silently replay a different
    // speculative schedule from the source runtime.
    for (const auto& sample : samples) {
        for (size_t index = 1; index < sample.size(); ++index) {
            const auto& previous = sample[index - 1];
            const auto& current = sample[index];
            if (previous.trace_version >= 2 && current.trace_version >= 2
                && (previous.target_kv_cache_tokens_after_decision != current.target_kv_cache_tokens
                    || previous.draft_kv_cache_tokens_after_decision != current.draft_kv_cache_tokens)) {
                throw std::runtime_error("PEARL v2 cache continuity error between iterations "
                    + std::to_string(previous.iteration) + " and " + std::to_string(current.iteration));
            }
        }
    }
    if (spec.sample_index >= 0 && static_cast<size_t>(spec.sample_index) >= samples.size()) {
        throw std::runtime_error(
            "Requested speculative sample_index=" + std::to_string(spec.sample_index)
            + " but control trace contains " + std::to_string(samples.size()) + " sample(s)");
    }

    const uint32_t first_sample = spec.sample_index < 0 ? 0 : static_cast<uint32_t>(spec.sample_index);
    const uint32_t last_sample = spec.sample_index < 0
        ? static_cast<uint32_t>(samples.size()) : first_sample + 1;
    for (uint32_t sample_id = first_sample; sample_id < last_sample; ++sample_id) {
        for (auto control : samples[sample_id]) {
            if (control.iteration < spec.skip_warmup_iterations) continue;
            if (spec.max_control_iterations != 0
                && speculative_control.size() >= spec.max_control_iterations) break;
            control.sample_index = sample_id;
            speculative_control.push_back(std::move(control));
        }
        if (spec.max_control_iterations != 0
            && speculative_control.size() >= spec.max_control_iterations) break;
    }
    if (speculative_control.empty()) {
        throw std::runtime_error("No speculative control records remain after warm-up filtering.");
    }
    std::cout << "[Speculative] control trace samples=" << samples.size()
              << ", selected sample="
              << (spec.sample_index < 0 ? std::string("all") : std::to_string(spec.sample_index))
              << ", control iterations=" << speculative_control.size() << "\n";
}

void TraceGenerator::add_host_trace(const std::string& layer, const std::vector<DeviceType>& r_targets, const std::vector<DeviceType>& w_targets)
{
    if (!r_targets.empty()) { add_memory_trace(TraceType::READ, layer, HOST, r_targets); }
    add_trace(TraceType::COMPUTE, layer, HOST);
    if (!w_targets.empty()) { add_memory_trace(TraceType::WRITE, layer, HOST, w_targets); }
}

void TraceGenerator::add_gemv_trace(const std::string& layer, const std::vector<std::string>& cmds, unsigned int next)
{
    std::unordered_map<std::string, unsigned int> tile_map = cfgs.get_layer_tile_map();
    if (tile_map.find(layer) == tile_map.end() || tile_map.at(layer) == 0) {
        throw std::runtime_error("Missing or zero PIM output_tile for layer: " + layer);
    }
    unsigned int tile_num = tile_map.at(layer);

    for (int i = 0; i < tile_num; i++) {
        for (const auto& cmd : cmds) {
            trace_queue.push_back(std::make_shared<PimTrace>(TraceType::PIM, layer, cmd, next));
        }
    }
    trace_queue.push_back(std::make_shared<PimTrace>(TraceType::PIM, layer, "addertree", next));
}

void TraceGenerator::add_rope_trace(const std::string& layer, const std::vector<std::string>& cmds)
{
    std::unordered_map<std::string, unsigned int> tile_map = cfgs.get_layer_tile_map();
    if (tile_map.find(layer) == tile_map.end() || tile_map.at(layer) == 0) {
        throw std::runtime_error("Missing or zero PIM output_tile for layer: " + layer);
    }
    unsigned int tile_num = tile_map.at(layer);

    for (int i = 0; i < tile_num; i++) {
        for (const auto& cmd : cmds) {
            trace_queue.push_back(std::make_shared<PimTrace>(TraceType::PIM, layer, cmd, PIM));
        }
    }
}
