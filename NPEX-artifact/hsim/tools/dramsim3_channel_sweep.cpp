#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "bridge.h"

namespace {

constexpr std::uint64_t kBaseAddress = 0x10000000ULL;
constexpr std::array<unsigned int, 4> kSweepChannels{1, 2, 3, 4};

struct Options {
    std::string memory_type{"LPDDR5"};
    std::string chip_config{"8Gb_x16"};
    std::string timing_preset{"6400"};
    std::uint32_t request_bytes{64};
    std::uint32_t calibration_requests{4096};
    double frequency_ghz{3.2};
    std::string channel_model{"native"};
    double channel_efficiency{1.0};
};

struct Measurement {
    double elapsed_ns{0.0};
};

struct DirectionProfile {
    double first_ns{0.0};
    double bandwidth_gbps{0.0};
    double validation_error_pct{0.0};
};

bool is_power_of_two(unsigned int value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

template <typename T>
T parse_number(std::string_view value, const char* option)
{
    T result{};
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), result);
    if (error != std::errc{} || end != value.data() + value.size()) {
        throw std::runtime_error(std::string("Invalid value for ") + option + ": " + std::string(value));
    }
    return result;
}

double parse_double(std::string_view value, const char* option)
{
    std::string owned(value);
    std::size_t parsed = 0;
    const double result = std::stod(owned, &parsed);
    if (parsed != owned.size() || !std::isfinite(result)) {
        throw std::runtime_error(std::string("Invalid value for ") + option + ": " + owned);
    }
    return result;
}

Options parse_options(int argc, char* argv[])
{
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string_view option(argv[index]);
        if (option == "--help") {
            std::cout
                << "Usage: dramsim3_channel_sweep [options]\n"
                << "  --memory-type <type>        default: LPDDR5\n"
                << "  --chip-config <name>        default: 8Gb_x16\n"
                << "  --timing-preset <rate>      default: 6400\n"
                << "  --request-bytes <bytes>     default: 64 (matches memory.json)\n"
                << "  --requests <count>          default: 4096\n"
                << "  --frequency-ghz <GHz>       default: 3.2\n"
                << "  --channel-model <mode>      native | linear_logical (default: native)\n"
                << "  --channel-efficiency <0..1> additional-channel efficiency; default: 1.0\n";
            std::exit(0);
        }
        if (index + 1 >= argc) {
            throw std::runtime_error("Missing value for " + std::string(option));
        }
        const std::string_view value(argv[++index]);
        if (option == "--memory-type") {
            options.memory_type = value;
        } else if (option == "--chip-config") {
            options.chip_config = value;
        } else if (option == "--timing-preset") {
            options.timing_preset = value;
        } else if (option == "--request-bytes") {
            options.request_bytes = parse_number<std::uint32_t>(value, "--request-bytes");
        } else if (option == "--requests") {
            options.calibration_requests = parse_number<std::uint32_t>(value, "--requests");
        } else if (option == "--frequency-ghz") {
            options.frequency_ghz = parse_double(value, "--frequency-ghz");
        } else if (option == "--channel-model") {
            options.channel_model = value;
        } else if (option == "--channel-efficiency") {
            options.channel_efficiency = parse_double(value, "--channel-efficiency");
        } else {
            throw std::runtime_error("Unknown option: " + std::string(option));
        }
    }

    if (options.request_bytes == 0 || options.calibration_requests < 2 || options.frequency_ghz <= 0.0) {
        throw std::runtime_error("request bytes/frequency must be positive and requests must be at least 2");
    }
    if (options.channel_model != "native" && options.channel_model != "linear_logical") {
        throw std::runtime_error("--channel-model must be native or linear_logical");
    }
    if (options.channel_efficiency <= 0.0 || options.channel_efficiency > 1.0) {
        throw std::runtime_error("--channel-efficiency must be in (0, 1]");
    }
    return options;
}

Measurement measure_stream(const std::string& config_path, const Options& options,
                           tlm::tlm_command command, std::uint64_t base_address,
                           std::uint32_t request_count)
{
    dramsim3::Bridge bridge(config_path.c_str(), DRAMSIM3_PATH);
    std::vector<std::unique_ptr<tlm::tlm_generic_payload>> requests;
    requests.reserve(request_count);

    for (std::uint32_t request_id = 0; request_id < request_count; ++request_id) {
        auto request = std::make_unique<tlm::tlm_generic_payload>();
        request->set_command(command);
        request->set_address(base_address
            + static_cast<std::uint64_t>(request_id) * options.request_bytes);
        request->set_data_length(options.request_bytes);
        bridge.sendCommand(*request);
        requests.push_back(std::move(request));
    }

    std::uint32_t completed = 0;
    std::uint64_t ticks = 0;
    constexpr std::uint64_t kMaxTicks = 10000000ULL;
    while (completed < request_count && ticks < kMaxTicks) {
        bridge.ClockTick();
        ++ticks;
        while (bridge.getCompletedCommand() != nullptr) {
            ++completed;
        }
    }
    if (completed != request_count) {
        throw std::runtime_error("DRAMsim3 did not complete the calibration stream");
    }
    return {static_cast<double>(ticks) / options.frequency_ghz};
}

DirectionProfile profile(const std::string& config_path, const Options& options,
                         tlm::tlm_command command)
{
    const auto first = measure_stream(config_path, options, command, kBaseAddress, 1);
    const auto train = measure_stream(config_path, options, command, kBaseAddress,
                                      options.calibration_requests);
    const double ns_per_request = std::max(
        1.0 / options.frequency_ghz,
        (train.elapsed_ns - first.elapsed_ns)
            / static_cast<double>(options.calibration_requests - 1));

    const auto validation = measure_stream(
        config_path, options, command, kBaseAddress + 0x01000000ULL,
        options.calibration_requests * 2U);
    const double prediction = first.elapsed_ns
        + static_cast<double>(options.calibration_requests * 2U - 1U) * ns_per_request;
    const double validation_error = 100.0 * (prediction - validation.elapsed_ns)
        / validation.elapsed_ns;

    return {first.elapsed_ns,
            static_cast<double>(options.request_bytes) / ns_per_request,
            validation_error};
}

std::string config_path_for(unsigned int channels, const Options& options)
{
    return std::string(DRAMSIM3_PATH) + "configs/" + options.memory_type + "/"
        + options.chip_config + "_" + options.timing_preset + "_ch"
        + std::to_string(channels) + ".ini";
}

void print_csv_header()
{
    std::cout << "channels,status,read_first_ns,read_bw_GBps,read_validation_error_pct,"
                 "write_first_ns,write_bw_GBps,write_validation_error_pct\n";
}

} // namespace

int sc_main(int argc, char* argv[])
{
    try {
        const Options options = parse_options(argc, argv);
        std::cout << "# DRAMsim3 saturated sequential-stream sweep; request_bytes="
                  << options.request_bytes << ", requests=" << options.calibration_requests
                  << ", frequency_GHz=" << options.frequency_ghz
                  << ", channel_model=" << options.channel_model
                  << ", additional_channel_efficiency=" << options.channel_efficiency << "\n";
        if (options.channel_model == "linear_logical") {
            std::cout << "# validation_error_pct applies to the one-channel DRAMsim3 reference; "
                         "multi-channel bandwidth uses the declared logical-channel efficiency model\n";
        }
        print_csv_header();

        DirectionProfile one_channel_read;
        DirectionProfile one_channel_write;
        if (options.channel_model == "linear_logical") {
            const std::string one_channel_path = config_path_for(1, options);
            if (!std::filesystem::exists(one_channel_path)) {
                throw std::runtime_error("Missing one-channel DRAMsim3 configuration: " + one_channel_path);
            }
            one_channel_read = profile(one_channel_path, options, tlm::TLM_READ_COMMAND);
            one_channel_write = profile(one_channel_path, options, tlm::TLM_WRITE_COMMAND);
        }

        for (const unsigned int channels : kSweepChannels) {
            if (options.channel_model == "linear_logical") {
                const double scale = 1.0 + options.channel_efficiency
                    * static_cast<double>(channels - 1U);
                std::cout << std::fixed << std::setprecision(3)
                          << channels << ",linear_logical,"
                          << one_channel_read.first_ns << ','
                          << one_channel_read.bandwidth_gbps * scale << ','
                          << one_channel_read.validation_error_pct << ','
                          << one_channel_write.first_ns << ','
                          << one_channel_write.bandwidth_gbps * scale << ','
                          << one_channel_write.validation_error_pct << "\n";
                continue;
            }
            const std::string config_path = config_path_for(channels, options);
            if (!std::filesystem::exists(config_path)) {
                std::cout << channels << ",missing_config,,,,,,,\n";
                continue;
            }
            if (!is_power_of_two(channels)) {
                // DRAMsim3 derives channel bits using floor(log2(channels)),
                // so an address can never select every controller for e.g. 3.
                std::cout << channels << ",invalid_non_power_of_two,,,,,,,\n";
                continue;
            }

            const DirectionProfile read = profile(config_path, options, tlm::TLM_READ_COMMAND);
            const DirectionProfile write = profile(config_path, options, tlm::TLM_WRITE_COMMAND);
            std::cout << std::fixed << std::setprecision(3)
                      << channels << ",ok,"
                      << read.first_ns << ',' << read.bandwidth_gbps << ','
                      << read.validation_error_pct << ','
                      << write.first_ns << ',' << write.bandwidth_gbps << ','
                      << write.validation_error_pct << "\n";
        }
    } catch (const std::exception& error) {
        std::cerr << "dramsim3_channel_sweep: " << error.what() << "\n";
        return 1;
    }
    return 0;
}
