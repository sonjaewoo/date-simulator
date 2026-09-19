#ifndef DRAM_MACRO_MODEL_H
#define DRAM_MACRO_MODEL_H

#include <cstdint>
#include <string>

#include <tlm.h>

#include <utilities/configurations.h>

// A fast-layer transfer is too large to decompose into all of its cache-line
// requests at simulation time.  This model measures the configured DRAMsim3
// instance with a short representative stream, then turns a tensor transfer
// into one resource-reserving macro request.  It deliberately retains the
// DRAM configuration's timing, address mapping, bank policy, and channel
// count in the calibration while avoiding O(tensor_bytes / cache_line) work.
class DramMacroModel {
public:
    DramMacroModel(std::string name, std::string dramsim_config_path,
                   std::string dramsim_root, double frequency_ghz,
                   uint32_t request_bytes, uint32_t logical_channels,
                   const SyntheticNPUInfo& npu_info);

    // Returns service time in ns.  Queueing between macro requests is handled
    // by the owning wrapper, which owns the physical memory resource.
    uint64_t estimate_ns(tlm::tlm_command command, uint64_t bytes) const;

    static bool is_macro_request(const tlm::tlm_generic_payload& trans);
    static uint64_t macro_bytes(const tlm::tlm_generic_payload& trans);

private:
    struct Measurement {
        double elapsed_ns{0.0};
        uint32_t completed{0};
    };

    Measurement measure_stream(const std::string& config_path,
                               tlm::tlm_command command, uint64_t base_address,
                               uint32_t request_count) const;
    void calibrate();
    void print_profile() const;
    std::string one_channel_config_path() const;
    double effective_channels() const;

    std::string name_;
    std::string dramsim_config_path_;
    std::string dramsim_root_;
    double tick_ns_{0.0};
    uint32_t request_bytes_{64};
    uint32_t logical_channels_{1};
    SyntheticNPUInfo npu_info_;

    double read_first_ns_{0.0};
    double write_first_ns_{0.0};
    double read_ns_per_request_{0.0};
    double write_ns_per_request_{0.0};
    double read_validation_error_pct_{0.0};
    double write_validation_error_pct_{0.0};
};

#endif
