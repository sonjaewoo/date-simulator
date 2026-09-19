#include "dram_macro_model.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "bridge.h"

namespace {

constexpr uint64_t kCalibrationBaseAddress = 0x10000000ULL;

} // namespace

DramMacroModel::DramMacroModel(std::string name, std::string dramsim_config_path,
                               std::string dramsim_root, double frequency_ghz,
                               uint32_t request_bytes, uint32_t logical_channels,
                               const SyntheticNPUInfo& npu_info)
    : name_(std::move(name)), dramsim_config_path_(std::move(dramsim_config_path)),
      dramsim_root_(std::move(dramsim_root)),
      tick_ns_(frequency_ghz > 0.0 ? 1.0 / frequency_ghz : 0.0),
      request_bytes_(std::max(1U, request_bytes)),
      logical_channels_(std::max(1U, logical_channels)), npu_info_(npu_info)
{
    if (tick_ns_ <= 0.0) {
        throw std::runtime_error("DRAM macro model requires a positive DRAM frequency");
    }
    if (npu_info_.fast_memory_backend == "dramsim3_calibrated") {
        calibrate();
    } else if (npu_info_.fast_memory_backend != "analytic") {
        throw std::runtime_error("Unsupported synthetic_npu.fast_memory_backend: "
                                 + npu_info_.fast_memory_backend);
    }
}

bool DramMacroModel::is_macro_request(const tlm::tlm_generic_payload& trans)
{
    return trans.get_pim_cmd() == "fast_memory";
}

uint64_t DramMacroModel::macro_bytes(const tlm::tlm_generic_payload& trans)
{
    return (static_cast<uint64_t>(trans.get_bst_size()) << 32U)
         | static_cast<uint64_t>(trans.get_data_length());
}

DramMacroModel::Measurement DramMacroModel::measure_stream(
    const std::string& config_path, tlm::tlm_command command,
    uint64_t base_address, uint32_t request_count) const
{
    dramsim3::Bridge bridge(config_path.c_str(), dramsim_root_.c_str());
    std::vector<std::unique_ptr<tlm::tlm_generic_payload>> requests;
    requests.reserve(request_count);

    for (uint32_t request_id = 0; request_id < request_count; ++request_id) {
        auto trans = std::make_unique<tlm::tlm_generic_payload>();
        trans->set_command(command);
        trans->set_address(base_address + static_cast<uint64_t>(request_id) * request_bytes_);
        trans->set_data_length(request_bytes_);
        bridge.sendCommand(*trans);
        requests.push_back(std::move(trans));
    }

    uint32_t completed = 0;
    uint64_t ticks = 0;
    // The bound protects against an invalid DRAMsim3 configuration turning a
    // calibration error into an infinite elaboration-time loop.
    const uint64_t max_ticks = 10000000ULL;
    while (completed < request_count && ticks < max_ticks) {
        bridge.ClockTick();
        ++ticks;
        while (bridge.getCompletedCommand() != nullptr) {
            ++completed;
        }
    }
    if (completed != request_count) {
        throw std::runtime_error("DRAMsim3 macro calibration did not complete");
    }
    return {static_cast<double>(ticks) * tick_ns_, completed};
}

std::string DramMacroModel::one_channel_config_path() const
{
    const std::size_t marker = dramsim_config_path_.rfind("_ch");
    const std::size_t extension = dramsim_config_path_.rfind(".ini");
    if (marker == std::string::npos || extension == std::string::npos
        || extension <= marker + 3) {
        throw std::runtime_error("Cannot derive one-channel DRAMsim3 configuration from: "
                                 + dramsim_config_path_);
    }
    const std::string channel_text = dramsim_config_path_.substr(marker + 3,
                                                                   extension - marker - 3);
    if (!std::all_of(channel_text.begin(), channel_text.end(), [](unsigned char value) {
            return std::isdigit(value) != 0;
        })) {
        throw std::runtime_error("Cannot derive one-channel DRAMsim3 configuration from: "
                                 + dramsim_config_path_);
    }
    return dramsim_config_path_.substr(0, marker) + "_ch1"
        + dramsim_config_path_.substr(extension);
}

double DramMacroModel::effective_channels() const
{
    if (npu_info_.fast_memory_channel_model != "linear_logical") {
        return static_cast<double>(logical_channels_);
    }
    return 1.0 + npu_info_.fast_memory_channel_efficiency
        * static_cast<double>(logical_channels_ - 1U);
}

void DramMacroModel::calibrate()
{
    const uint32_t train_requests = std::max(2U, npu_info_.dram_macro_calibration_requests);
    const uint32_t validation_requests = train_requests * 2U;
    const bool linear_logical = npu_info_.fast_memory_channel_model == "linear_logical";
    const std::string calibration_config = linear_logical
        ? one_channel_config_path() : dramsim_config_path_;

    const auto fit = [&](tlm::tlm_command command, double& first_ns,
                         double& ns_per_request, double& validation_error_pct) {
        const auto first = measure_stream(calibration_config, command, kCalibrationBaseAddress, 1);
        const auto train = measure_stream(calibration_config, command, kCalibrationBaseAddress,
                                          train_requests);
        first_ns = first.elapsed_ns;
        ns_per_request = std::max(
            tick_ns_, (train.elapsed_ns - first_ns) / static_cast<double>(train_requests - 1));

        const auto validation = measure_stream(calibration_config, command,
            kCalibrationBaseAddress + 0x01000000ULL, validation_requests);
        const double prediction = first_ns
            + static_cast<double>(validation_requests - 1) * ns_per_request;
        validation_error_pct = validation.elapsed_ns > 0.0
            ? 100.0 * (prediction - validation.elapsed_ns) / validation.elapsed_ns
            : 0.0;
        if (linear_logical) {
            ns_per_request /= effective_channels();
        }
    };

    fit(tlm::TLM_READ_COMMAND, read_first_ns_, read_ns_per_request_,
        read_validation_error_pct_);
    fit(tlm::TLM_WRITE_COMMAND, write_first_ns_, write_ns_per_request_,
        write_validation_error_pct_);
    print_profile();
}

uint64_t DramMacroModel::estimate_ns(tlm::tlm_command command, uint64_t bytes) const
{
    if (bytes == 0) return 0;
    if (npu_info_.fast_memory_backend == "analytic") {
        return npu_info_.fast_memory_fixed_latency_ns + static_cast<uint64_t>(std::ceil(
            static_cast<double>(bytes) / npu_info_.fast_memory_bandwidth_GBps));
    }

    const uint64_t requests = (bytes + request_bytes_ - 1) / request_bytes_;
    const bool is_write = command == tlm::TLM_WRITE_COMMAND;
    const double first_ns = is_write ? write_first_ns_ : read_first_ns_;
    const double ns_per_request = is_write ? write_ns_per_request_ : read_ns_per_request_;
    const double raw_ns = first_ns
        + static_cast<double>(requests > 0 ? requests - 1 : 0) * ns_per_request;
    return npu_info_.fast_memory_fixed_latency_ns
        + static_cast<uint64_t>(std::ceil(raw_ns));
}

void DramMacroModel::print_profile() const
{
    if (npu_info_.fast_memory_backend != "dramsim3_calibrated") return;
    const auto bandwidth = [&](double ns_per_request) {
        return static_cast<double>(request_bytes_) / ns_per_request;
    };
    std::cout << std::fixed << std::setprecision(3)
              << "[DRAMMacroCalibration] " << name_
              << " samples=" << npu_info_.dram_macro_calibration_requests
              << " channel_model=" << npu_info_.fast_memory_channel_model
              << " logical_channels=" << logical_channels_
              << " effective_channels=" << effective_channels()
              << " additional_channel_efficiency="
              << npu_info_.fast_memory_channel_efficiency
              << (npu_info_.fast_memory_channel_model == "linear_logical"
                      ? " validation_scope=one_channel"
                      : "")
              << " read_first_ns=" << read_first_ns_
              << " read_bw_GBps=" << bandwidth(read_ns_per_request_)
              << " read_validation_error_pct=" << read_validation_error_pct_
              << " write_first_ns=" << write_first_ns_
              << " write_bw_GBps=" << bandwidth(write_ns_per_request_)
              << " write_validation_error_pct=" << write_validation_error_pct_
              << "\n";
}
