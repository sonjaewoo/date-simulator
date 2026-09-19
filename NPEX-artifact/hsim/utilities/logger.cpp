#include "logger.h"

void Logger::update_start(const std::string& layer, DeviceType type)
{
    update_start_at(layer, type, static_cast<int64_t>(sc_core::sc_time_stamp().to_double() / 1000));
}

void Logger::update_end(const std::string& layer, DeviceType type)
{
    update_end_at(layer, type, static_cast<int64_t>(sc_core::sc_time_stamp().to_double() / 1000));
}

void Logger::update_start_at(const std::string& layer, DeviceType type, int64_t cycle)
{
    switch (type) {
        case DeviceType::HOST:
            if (host_info.find(layer) == host_info.end())
                host_info[layer].first = cycle;
            break;
        case DeviceType::NPU:
            if (npu_info.find(layer) == npu_info.end())
                npu_info[layer].start = cycle;
            break;
        case DeviceType::PIM:
            if (pim_info.find(layer) == pim_info.end())
                pim_info[layer].first = cycle;
            break;
        case DeviceType::DRAM:
            if (dram_info.find(layer) == dram_info.end())
                dram_info[layer].first = cycle;
            break;            
    }
}

void Logger::update_end_at(const std::string& layer, DeviceType type, int64_t cycle)
{
    switch (type) {
        case DeviceType::HOST:
            host_info[layer].second = cycle;
            break;
        case DeviceType::PIM:
            pim_info[layer].second = cycle;
            break;
        case DeviceType::DRAM:
            dram_info[layer].second = cycle;
            break;            
    }
}

void Logger::update_npu_start(const std::string& layer, int64_t start)
{
    int64_t cycle = static_cast<int64_t>(sc_core::sc_time_stamp().to_double() / 1000);
    auto& cycle_info = npu_info[layer];

	if (layer == "D_MHA_out") {
        cycle_info.start = start;
	}
	
	else {
	    if (cycle_info.start == -1) {
        	if (layer == "D_Q0")
    	        cycle_info.start = cycle;
	        else {
            	cycle_info.start = start;
        	}
    	}
	}
}

void Logger::update_npu_end(const std::string& layer, int64_t end, int64_t compute_cycle)
{
    npu_info[layer].end = end;
    npu_info[layer].compute_cycle = compute_cycle;
}

void Logger::print_and_save_log(const std::string& filename)
{
    std::ostream* out_stream = &std::cout;
    std::ofstream ofs;

    if (!filename.empty()) {
        ofs.open(filename);
        if (!ofs.is_open()) {
            std::cerr << "Failed to open file: " << filename << "\n";
            return;
        }
        out_stream = &ofs;
    }

    std::ostream& out = *out_stream;

    print_header(out, "Host Execution Cycle Info", {"Layer", "Start", "End", "Total"});
    for (const auto& [layer, cycle] : host_info) {
        if (cycle.first == -1) continue;
        print_cycle_data(out, layer, cycle.first, cycle.second);
    }

    print_header(out, "NPU Execution Cycle Info", {"Layer", "Start", "End", "DRAM delay", "Total"});
    for (const auto& [layer, cycle] : npu_info) {
        print_cycle_data(out, layer, cycle.start, cycle.end, cycle.compute_cycle);
    }

    if (cfgs.pim_enabled()) {
        print_header(out, "PIM Execution Cycle Info", {"Layer", "Start", "End", "Total"});
        for (const auto& [layer, cycle] : pim_info) {
            print_cycle_data(out, layer, cycle.first, cycle.second);
        }
    }

    if (cfgs.dram_enabled()) {
        print_header(out, "DRAM Execution Cycle Info", {"Layer", "Start", "End", "Total"});
        for (const auto& [layer, cycle] : dram_info) {
            print_cycle_data(out, layer, cycle.first, cycle.second);
        }
    }

    if (ofs.is_open()) {
        std::cout << "Cycle info saved to file: " << filename << "\n";
    }
}

void Logger::print_header(std::ostream& out, const std::string& title, const std::vector<std::string>& columns)
{
    out << "\n===== " << title << " =====\n";
    for (const auto& col : columns)
        out << col << " ";
    out << "\n" << std::string(60, '-') << "\n";
}


void Logger::print_cycle_data(std::ostream& out, const std::string& layer, int64_t start, int64_t end, int64_t extra)
{
    // if (layer != "D_up_proj" && layer != "D_gate_proj" &&
    //     layer != "D_down_proj" && layer != "Mul1" && layer != "D_MHA_out" &&
    //     layer != "RMSNorm_D_post_mlp" && layer != "D_residual_conn2") {
    //     int duration = (end > 0) ? (end - start) : -1;
    //     out << layer << " "
    //         << start << " "
    //         << (end > 0 ? std::to_string(end) : "N/A") << " ";
    //     if (extra >= 0) {
    //         out << duration-extra << " ";
    //     }
    //     out << (duration >= 0 ? std::to_string(duration) : "N/A") << "\n";
    // }
    
    int64_t duration = (end > 0) ? (end - start) : -1;
        out << layer << " "
            << start << " "
            << (end > 0 ? std::to_string(end) : "N/A") << " ";
        if (extra >= 0) {
            out << duration-extra << " ";
        }
        out << (duration >= 0 ? std::to_string(duration) : "N/A") << "\n";    
}
