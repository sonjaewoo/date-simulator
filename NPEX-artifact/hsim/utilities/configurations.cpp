#include "configurations.h"

#include <cstdlib>

Configurations::Configurations()
{};

void Configurations::init_configurations()
{
    init_system();
    if (!speculative_enabled() || !synthetic_npu_enabled()) {
        throw std::runtime_error(
            "This artifact supports speculative replay with synthetic_npu.enabled=true only");
    }
    init_dram();
}

void Configurations::print_configurations()
{
    std::cout << "\n========== Simulation Configuration ==========\n";

    std::cout << "Frequency (GHz):\n";
    std::cout << "  ∙ Host (BP) : " << host_frequency << "\n";

    std::cout << "Components:\n";
    std::cout << "  ∙ Enable DRAM : " << (enable_dram ? "Yes" : "No") << "\n";
    std::cout << "  ∙ Enable PIM  : " << (enable_pim  ? "Yes" : "No") << "\n";
    std::cout << "  ∙ Synthetic NPU trace : " << (synthetic_npu.enabled ? "Yes" : "No") << "\n";
    std::cout << "  ∙ Synthetic PIM trace : " << (synthetic_pim.enabled ? "Yes" : "No") << "\n";
    if (fast_layer_mode_enabled()) {
        std::cout << "  ∙ Fast-layer record granularity : NPU=" << synthetic_npu.fast_layer_granularity
                  << ", PIM=" << synthetic_pim.fast_layer_granularity << "\n";
        std::cout << "  ∙ Fast memory backend : " << synthetic_npu.fast_memory_backend;
        if (synthetic_npu.fast_memory_backend == "dramsim3_calibrated") {
            std::cout << " (DRAMsim3 calibration requests="
                      << synthetic_npu.dram_macro_calibration_requests
                      << ", channel model=" << synthetic_npu.fast_memory_channel_model;
            if (synthetic_npu.fast_memory_channel_model == "linear_logical") {
                std::cout << ", additional-channel efficiency="
                          << synthetic_npu.fast_memory_channel_efficiency;
            }
            std::cout << ")";
        }
        std::cout << "\n";
    }

    std::cout << "\nMapping Scenario:\n";
    std::cout << "  ∙ Execution mode : " << execution_mode << "\n";
    std::cout << "  ∙ Scenario ID : " << (speculative_enabled() ? speculative.scenario_id : scenario_id) << "\n";

    std::cout << "\nDRAM-PIM Structure: " << memory_structure << "\n";

    std::cout << "\nDRAM Configuration:\n";
    if (enable_dram) {
        std::cout << "  ∙ Type              : " << draminfo.type << "_" << draminfo.config << "\n";
        std::cout << "  ∙ Backend           : " << draminfo.backend << "\n";
        std::cout << "  ∙ Timing Preset     : " << draminfo.preset << "\n";
        std::cout << "  ∙ Channels          : " << draminfo.channels << "\n";
        std::cout << "  ∙ Capacity (GB)     : " << draminfo.capacity << "\n";
        std::cout << "  ∙ Frequency (GHz)   : " << draminfo.freq << "\n";
    } else {
        std::cout << "  ∙ [Disabled]\n";
    }

    std::cout << "\nPIM Configuration:\n";
    if (enable_pim) {
        std::cout << "  ∙ Type              : " << piminfo.type << "_" << piminfo.config << "\n";
        std::cout << "  ∙ Backend           : " << piminfo.backend << "\n";
        std::cout << "  ∙ Timing Preset     : " << piminfo.preset << "\n";
        std::cout << "  ∙ Channels          : " << piminfo.channels << "\n";
        std::cout << "  ∙ Capacity (GB)     : " << piminfo.capacity << "\n";
        std::cout << "  ∙ Frequency (GHz)   : " << piminfo.freq << "\n";
    } else {
        std::cout << "  [Disabled]\n";
    }
    std::cout << "\n Workload:\n";
    std::cout << "  ∙ Input Sequence Length  : " << input_sequence_length << "\n";
    std::cout << "  ∙ Output Sequence Length : " << output_sequence_length << "\n";
    std::cout << "  ∙ Decoder Block : " << decoder_block << "\n";
    std::cout << "==============================================\n\n";
}

void Configurations::init_system()
{
    Json::Value root = parse_json("system");
    const Json::Value arch = root["architecture"];
    const Json::Value work = root["workload"];
    const Json::Value freq = root["frequency"];

    memory_structure = arch["memory_structure"].asString();

    const Json::Value synthetic = root["synthetic_npu"];
    if (!synthetic.isNull()) {
        synthetic_npu.enabled = synthetic.get("enabled", false).asBool();
        synthetic_npu.compute_ns_per_layer = synthetic.get("compute_ns_per_layer", 0).asUInt();
        synthetic_npu.weight_read_bytes_per_layer = synthetic.get("weight_read_bytes_per_layer", 0).asUInt();
        synthetic_npu.output_write_bytes_per_layer = synthetic.get("output_write_bytes_per_layer", 0).asUInt();
        synthetic_npu.max_inflight_requests = synthetic.get("max_inflight_requests", 256).asUInt();
        synthetic_npu.model_profile = synthetic.get("model_profile", "").asString();
        synthetic_npu.fast_layer_mode = synthetic.get("fast_layer_mode", false).asBool();
        synthetic_npu.fast_layer_granularity =
            synthetic.get("fast_layer_granularity", "macro").asString();
        synthetic_npu.fast_memory_backend =
            synthetic.get("fast_memory_backend", "analytic").asString();
        synthetic_npu.fast_memory_channel_model =
            synthetic.get("fast_memory_channel_model", "native").asString();
        synthetic_npu.fast_memory_channel_efficiency =
            synthetic.get("fast_memory_channel_efficiency", 1.0).asDouble();
        synthetic_npu.dram_macro_calibration_requests =
            synthetic.get("dram_macro_calibration_requests", 4096).asUInt();
        synthetic_npu.fast_memory_bandwidth_GBps =
            synthetic.get("fast_memory_bandwidth_GBps", 25.6).asDouble();
        synthetic_npu.fast_memory_fixed_latency_ns =
            synthetic.get("fast_memory_fixed_latency_ns", 100).asUInt();
        synthetic_npu.analytical_compute_enabled =
            synthetic.get("analytical_compute_enabled", true).asBool();
        synthetic_npu.macs_per_cycle = synthetic.get("macs_per_cycle", 1024).asUInt();
        synthetic_npu.compute_frequency_GHz =
            synthetic.get("compute_frequency_GHz", 0.5).asDouble();
        if (synthetic_npu.macs_per_cycle == 0 || synthetic_npu.compute_frequency_GHz <= 0.0) {
            throw std::runtime_error("synthetic_npu MAC throughput and frequency must be positive");
        }
    }
    const Json::Value synthetic_pim_config = root["synthetic_pim"];
    if (!synthetic_pim_config.isNull()) {
        synthetic_pim.enabled = synthetic_pim_config.get("enabled", false).asBool();
        synthetic_pim.layer_count = synthetic_pim_config.get("layer_count", 0).asUInt();
        synthetic_pim.compute_ns_per_layer = synthetic_pim_config.get("compute_ns_per_layer", 0).asUInt();
        synthetic_pim.commands_per_layer = synthetic_pim_config.get("commands_per_layer", 1).asUInt();
        synthetic_pim.command = synthetic_pim_config.get("command", "mac").asString();
        synthetic_pim.model_profile = synthetic_pim_config.get("model_profile", "").asString();
        synthetic_pim.aggregate_micro_commands =
            synthetic_pim_config.get("aggregate_micro_commands", false).asBool();
        synthetic_pim.fast_layer_mode = synthetic_pim_config.get("fast_layer_mode", false).asBool();
        synthetic_pim.fast_layer_granularity =
            synthetic_pim_config.get("fast_layer_granularity", "macro").asString();
    }
    for (const auto& [who, granularity] : {
             std::pair<const char*, const std::string&>{"synthetic_npu", synthetic_npu.fast_layer_granularity},
             std::pair<const char*, const std::string&>{"synthetic_pim", synthetic_pim.fast_layer_granularity}}) {
        if (granularity != "macro" && granularity != "layer") {
            throw std::runtime_error(std::string(who) + ".fast_layer_granularity must be "
                                     "\"macro\" or \"layer\", got: " + granularity);
        }
    }

    target_model = work["model"].asString();
    execution_mode = work.get("execution_mode", "speculative").asString();
    if (execution_mode != "speculative") {
        throw std::runtime_error("This artifact supports execution_mode=speculative only");
    }
    num_attn_heads  = work["attention_head"].asUInt();
    scenario_id = work["scenario_id"].asUInt();
    input_sequence_length = work["input_sequence_length"].asUInt();
    output_sequence_length = work["output_sequence_length"].asUInt();
    decoder_block = work["decoder_block"].asUInt();

    const Json::Value spec = root["speculative"];
    if (!spec.isNull()) {
        speculative.scenario_id = spec.get("scenario_id", 0).asUInt();
        speculative.control_trace_file = spec.get("control_trace_file", "").asString();
        if (!speculative.control_trace_file.empty()) {
            const std::filesystem::path trace_path(speculative.control_trace_file);
            if (trace_path.is_relative()) {
                speculative.control_trace_file =
                    (std::filesystem::path(ROOT_PATH) / trace_path).lexically_normal().string();
            }
        }
        speculative.sample_index = spec.get("sample_index", -1).asInt();
        if (speculative.sample_index < -1) {
            throw std::runtime_error("speculative.sample_index must be -1 or non-negative");
        }
        speculative.skip_warmup_iterations = spec.get("skip_warmup_iterations", 0).asUInt();
        speculative.max_control_iterations = spec.get("max_control_iterations", 0).asUInt();
        speculative.exclude_prefill = spec.get("exclude_prefill", false).asBool();
        speculative.draft_profile_layer = spec.get("draft_profile_layer", "D_K0").asString();
    }
    if (speculative_enabled() && speculative.control_trace_file.empty()) {
        throw std::runtime_error("speculative.control_trace_file is required in speculative mode");
    }
    if (speculative_enabled() && !synthetic_npu.enabled) {
        throw std::runtime_error("speculative mode requires synthetic_npu.enabled=true");
    }
    if (synthetic_npu.enabled && synthetic_npu.max_inflight_requests == 0) {
        throw std::runtime_error("synthetic_npu.max_inflight_requests must be non-zero");
    }
    if (synthetic_npu.fast_layer_mode && synthetic_npu.fast_memory_bandwidth_GBps <= 0.0) {
        throw std::runtime_error("synthetic_npu.fast_memory_bandwidth_GBps must be positive");
    }
    if (synthetic_npu.fast_memory_backend != "analytic"
        && synthetic_npu.fast_memory_backend != "dramsim3_calibrated") {
        throw std::runtime_error("synthetic_npu.fast_memory_backend must be \"analytic\" or "
                                 "\"dramsim3_calibrated\"");
    }
    if (synthetic_npu.fast_memory_channel_model != "native"
        && synthetic_npu.fast_memory_channel_model != "linear_logical") {
        throw std::runtime_error("synthetic_npu.fast_memory_channel_model must be \"native\" or "
                                 "\"linear_logical\"");
    }
    if (synthetic_npu.fast_memory_channel_efficiency <= 0.0
        || synthetic_npu.fast_memory_channel_efficiency > 1.0) {
        throw std::runtime_error("synthetic_npu.fast_memory_channel_efficiency must be in (0, 1]");
    }
    if (synthetic_npu.fast_memory_backend == "dramsim3_calibrated"
        && synthetic_npu.dram_macro_calibration_requests < 2) {
        throw std::runtime_error("synthetic_npu.dram_macro_calibration_requests must be at least 2");
    }
    if (speculative_enabled()) {
        if (synthetic_npu.model_profile.empty() || synthetic_pim.model_profile.empty()) {
            throw std::runtime_error("speculative mode requires synthetic_npu/pim model_profile");
        }
        target_layer_profile = load_model_layer_profile(synthetic_npu.model_profile);
        draft_layer_profile = load_model_layer_profile(synthetic_pim.model_profile);
    }

    host_frequency = freq["host"].asDouble();

    cycle_log_file = root["cycle_log_file"].asString();
    performance_stats_file = root.get("performance_stats_file", "").asString();
    if (const char* override = std::getenv("HSIM_CYCLE_LOG_FILE");
        override != nullptr && override[0] != '\0') {
        cycle_log_file = override;
    }
    if (const char* override = std::getenv("HSIM_PERFORMANCE_STATS_FILE");
        override != nullptr && override[0] != '\0') {
        performance_stats_file = override;
    }

    if (memory_structure == "baseline") {
        enable_dram = true;
        enable_pim = false;
    } else if (memory_structure == "partitioned") {
        enable_dram = true;
        enable_pim = true;
    } else if (memory_structure == "unified") {
        enable_dram = false;
        enable_pim = true;
    } else {
        throw std::runtime_error("Invalid memory_structure: " + memory_structure);
    }
}

ModelLayerProfile Configurations::load_model_layer_profile(const std::string& file_name)
{
    const Json::Value root = parse_json(file_name);
    ModelLayerProfile profile;
    profile.name = root.get("name", file_name).asString();
    profile.layer_count = root["decoder_layers"].asUInt();
    profile.output_write_bytes = root.get("output_write_bytes", 0).asUInt64();
    profile.kv_cache_bytes_per_token = root.get("kv_cache_bytes_per_token", 0).asUInt64();
    profile.attention_context_tile_tokens = root.get("attention_context_tile_tokens", 128).asUInt();
    if (profile.layer_count == 0 || !root["operations"].isArray()) {
        throw std::runtime_error("Invalid model layer profile: " + file_name);
    }
    for (const auto& op : root["operations"]) {
        ModelOperationInfo entry;
        entry.name = op["name"].asString();
        entry.weight_read_bytes = op.get("weight_read_bytes", 0).asUInt64();
        entry.compute_share = op.get("compute_share", 1).asUInt();
        entry.macs_per_token = op.get("macs_per_token", 0).asUInt64();
        entry.macs_per_kv_token = op.get("macs_per_kv_token", 0).asUInt64();
        if (entry.name.empty() || entry.compute_share == 0) {
            throw std::runtime_error("Invalid operation in model profile: " + file_name);
        }
        profile.operations.push_back(entry);
    }
    if (profile.operations.empty()) {
        throw std::runtime_error("Model profile has no operations: " + file_name);
    }
    return profile;
}

void Configurations::init_dram()
{
    Json::Value root = parse_json("memory");

    dram_req_size = root["request_size"].asUInt(); /* Bytes */

    const Json::Value dram = root["dram"];
    draminfo.backend = dram["backend"].asString();
    draminfo.type = dram["memory_type"].asString();
    draminfo.config = dram["chip_config"].asString();
    draminfo.preset = dram["timing_preset"].asString();
    draminfo.channels = dram["num_channels"].asUInt();
    draminfo.capacity = dram["capacity"].asUInt();
    draminfo.freq = dram["clock_freq"].asDouble();

    const Json::Value pim = root["pim"];
    piminfo.backend = pim["backend"].asString();
    piminfo.type = pim["memory_type"].asString();
    piminfo.config = pim["chip_config"].asString();
    piminfo.preset = pim["timing_preset"].asString();    
    piminfo.channels = pim["num_channels"].asUInt();
    piminfo.capacity = pim["capacity"].asUInt();
    piminfo.freq = pim["clock_freq"].asDouble();

    init_host_profile();
    init_pim_profile(piminfo.type);
}

void Configurations::init_host_profile()
{
    Json::Value root = parse_json("profile/host");
    const Json::Value& host = root["host"];
    for (const auto& op : host.getMemberNames()) {
        host_profile[op] = host[op].asUInt();
    }
}

void Configurations::init_pim_profile(const std::string& pim_type)
{
    init_host_profile();
    
    /* PIM */
    std::string file_name = "profile/" + pim_type;
    
    if (input_sequence_length == 1024) {
        file_name += "_1024";   
    } else if (input_sequence_length == 512) {
        file_name += "_512";
    }

    if (target_model == "gemma2-9B") {
        file_name += "_9b";
    }

    Json::Value root = parse_json(file_name);
    
    const Json::Value mode_delay = root["mode_change_delay"];
    pim_to_dram_delay = mode_delay["PIM_to_DRAM"].asUInt();
    dram_to_pim_delay = mode_delay["DRAM_to_PIM"].asUInt();

    mode_change_delay = std::make_tuple(pim_to_dram_delay, dram_to_pim_delay);
    
    const Json::Value& operations = root["operations"];
    const Json::Value& layers = root["layers"];
    const Json::Value& channel_scaling = root["channel_scaling"];

    for (const auto& name : layers.getMemberNames()) {
        std::string operation = layers[name].asString();
        const Json::Value& cmd_array = operations[operation];
        PIMCmdProfile profile;

        // Amdahl-style channel model: serial work remains fixed while the
        // parallel fraction benefits from imperfect channel scaling.
        const Json::Value scaling = !channel_scaling.isNull()
            ? (channel_scaling.isMember(operation) ? channel_scaling[operation]
                                                    : channel_scaling["default"])
            : Json::Value();
        if (!scaling.isNull()) {
            const double parallel_fraction = scaling.get("parallel_fraction", 0.0).asDouble();
            const double efficiency = scaling.get("efficiency", 1.0).asDouble();
            const uint32_t saturation_channels = scaling.get("saturation_channels", 1).asUInt();
            if (parallel_fraction < 0.0 || parallel_fraction > 1.0
                || efficiency < 0.0 || efficiency > 1.0 || saturation_channels == 0) {
                throw std::runtime_error("Invalid channel_scaling for PIM operation: " + operation);
            }
            const uint32_t active_channels = std::min(piminfo.channels, saturation_channels);
            const double effective_channels = 1.0 + efficiency * (active_channels - 1);
            profile.channel_latency_scale = (1.0 - parallel_fraction)
                                          + parallel_fraction / effective_channels;
        }

        for (const auto& entry : cmd_array) {
            if (entry.isMember("output_tile")) {
                if (!entry["output_tile"].isUInt()) {
                    throw std::runtime_error("Invalid 'output_tile' value in operation '" + operation + "'");
                }
                layer_tile_map[name] = entry["output_tile"].asUInt();
                continue;
            }
            if (!entry.isMember("name") || !entry["name"].isString()) {
                throw std::runtime_error("Missing or invalid 'name' in operation '" + operation + "'");
            }
            if (!entry.isMember("latency") || !entry["latency"].isUInt()) {
                throw std::runtime_error("Missing or invalid 'latency' in operation '" + operation + "'");
            }
            if (!entry.isMember("count") || !entry["count"].isUInt()) {
                throw std::runtime_error("Missing or invalid 'count' in operation '" + operation + "'");
            }
            std::string cmd = entry["name"].asString();
            profile.cmd_map[cmd] = { entry["latency"].asUInt(), entry["count"].asUInt() };
            profile.ordered_cmd_name.push_back(cmd);
        }
        pim_profile[name] = profile;
    }
}

void Configurations::init_compiler()
{
    Json::Value compiler = parse_json("compiler");

    compileinfo.quantized = compiler["quantized"].asBool();
    compileinfo.additional_flags = compiler["additional_flags"].asString();

    compileinfo.midap_compiler = compiler["layer_compiler"].asString();
    if ((compileinfo.midap_compiler != "MIN_DRAM_ACCESS")
        && (compileinfo.midap_compiler != "HIDE_DRAM_LATENCY") && (compileinfo.midap_compiler != "DOUBLE_BUFFER")) {
        std::cout << "Invalid compiler option - " << compileinfo.midap_compiler << std::endl;
        exit(-1);
    }

    compileinfo.packet_size = compiler["packet_size"].asUInt();
    compileinfo.midap_level = compiler["midap_level"].asUInt();
    compileinfo.fmem_bank_num = compiler["fmem_bank_num"].asUInt();
    compileinfo.fmem_bank_size = compiler["fmem_bank_size"].asUInt();
    compileinfo.cim_num = compiler["cim_num"].asUInt();
    compileinfo.wmem_size = compiler["wmem_size"].asUInt();
    compileinfo.ewmem_size = compiler["ewmem_size"].asUInt();

    const Json::Value network = compiler["network"];
    auto it = network.begin();

    if (it->isObject()) {
        compileinfo.net_info.name = (*it)["net_name"].asString();
        std::string shared_offset = (*it)["shared_offset"].asString();
        std::string input_offset = (*it)["input_offset"].asString();
        std::string output_offset = (*it)["output_offset"].asString();
        std::string wb_offset = (*it)["wb_offset"].asString();
        std::string buf_offset = (*it)["buf_offset"].asString();
        std::stringstream shared_off_stream(shared_offset);
        std::stringstream input_off_stream(input_offset);
        std::stringstream output_off_stream(output_offset);
        std::stringstream wb_off_stream(wb_offset);
        std::stringstream buf_off_stream(buf_offset);

        shared_off_stream >> std::hex >> compileinfo.net_info.meminfo.shared_offset;
        input_off_stream >> std::hex >> compileinfo.net_info.meminfo.input_offset;
        output_off_stream >> std::hex >> compileinfo.net_info.meminfo.output_offset;
        wb_off_stream >> std::hex >> compileinfo.net_info.meminfo.wb_offset;
        buf_off_stream >> std::hex >> compileinfo.net_info.meminfo.buf_offset;

        std::stringstream temp;
        compileinfo.net_info.out_path = (*it)["path"].asString();
        temp << ROOT_PATH << compileinfo.net_info.out_path << "/";
        compileinfo.net_info.out_path = temp.str();

        compileinfo.net_info.prefix = (*it)["prefix"].asString();
        compileinfo.net_info.precompiled = (*it)["precompiled"].asBool();
    } 
}

void Configurations::compile_network()
{
    throw std::runtime_error("Network compilation is not included in this artifact");
}

void Configurations::init_sync_wait_map()
{
    NetworkInfo netinfo = get_netinfo();
    std::string base_path = netinfo.out_path + netinfo.prefix + "/core_1/";
    static const std::array<std::pair<std::string, SyncMap&>, 2> sync_maps = {{
        {"cpu", host_sync_map},
        {"pim", pim_sync_map}
    }};

    static const std::array<std::pair<std::string, WaitMap&>, 2> wait_maps = {{
        {"cpu", host_wait_map},
        {"pim", pim_wait_map}
    }};

    for (const auto& [device_name, sync_map] : sync_maps) {
        load_sync_wait_info(base_path + device_name + "_sync_info.txt", sync_map);
    }
    
    for (const auto& [device_name, wait_map] : wait_maps) {
        load_sync_wait_info(base_path + device_name + "_wait_info.txt", wait_map);
    }
    
    std::cout << "[HOST_SYNC_MAP]:\n";
    for (const auto& [key, value] : host_sync_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size(B): " << size
                  << " | Address: " << address << "\n";
    }
    std::cout << "[HOST_WAIT_MAP]:\n";
    for (const auto& [key, value] : host_wait_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size(B): " << size
                  << " | Address: " << address << "\n";
    }
    std::cout << "[PIM_SYNC_MAP]:\n";
    for (const auto& [key, value] : pim_sync_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;
        pim_sync_id_map[id] = true;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size: " << size
                  << " | Address: " <<  address << "\n";
    }
    std::cout << "[PIM_WAIT_MAP]:\n";
    for (const auto& [key, value] : pim_wait_map) {
        unsigned int id, size;
        uint64_t address;
        
        std::tie(id, size, address) = value;
        pim_wait_id_map[id] = true;

        std::cout << "Key: " << key
                  << " | ID: " << id
                  << " | Size: " << size
                  << " | Address: " << address << "\n";
    }
}

void Configurations::load_sync_wait_info(const std::string& file_path, BaseMap& info_map) {
    std::ifstream file(file_path);
    if (!file) {
        std::cerr << "Error: Cannot open file " << file_path << "\n";
        return;
    }

    std::string line;

    while (getline(file, line)) {
        std::istringstream ss(line);
        std::string layer;
        unsigned int id, dim;
        uint64_t mem_id, offset;

        if (!(ss >> id) || ss.get() != ',' ||
            !(ss >> mem_id) || ss.get() != ',' ||
            !(ss >> offset) || ss.get() != ',' ||
            !(ss >> dim) || ss.get() != ',' ||
            !getline(ss, layer)) {
            std::cerr << "Warning: Invalid line format -> " << line << "\n";
            continue;
        }

        uint64_t base_address = (mem_id == 1)
                                ? compileinfo.net_info.meminfo.input_offset
                                : compileinfo.net_info.meminfo.output_offset;
                                
        info_map.insert({layer, {id, dim*2, base_address + offset}}); // FP16(2B)
    }
}

Json::Value Configurations::parse_json(const std::string& file_name)
{
    // Keep the default configuration files immutable while allowing reproducible
    // experiment-specific system/memory configurations.
    const char* override_path = nullptr;
    if (file_name == "system") {
        override_path = std::getenv("HSIM_SYSTEM_CONFIG");
    } else if (file_name == "memory") {
        override_path = std::getenv("HSIM_MEMORY_CONFIG");
    }
    const std::string filePath = (override_path != nullptr && override_path[0] != '\0')
        ? std::string(override_path)
        : std::string(ROOT_PATH) + "hsim/configs/" + file_name + ".json";

    std::ifstream ifs(filePath);
    if(!ifs) {
        std::cerr << filePath << " file doesn't exist.\n";
        exit(EXIT_FAILURE);
    }

    std::string rawJson;

    ifs.seekg(0, std::ios::end);
    rawJson.reserve(ifs.tellg());
    ifs.seekg(0, std::ios::beg);

    rawJson.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

    JSONCPP_STRING err;
    Json::Value root;
    Json::CharReaderBuilder builder;
    const std::unique_ptr<Json::CharReader> reader(builder.newCharReader());
    if (!reader->parse(rawJson.c_str(), rawJson.c_str() + rawJson.length(), &root, &err)) {
        throw std::runtime_error("Invalid JSON in " + filePath + ": " + err);
    }

    return root;
}
