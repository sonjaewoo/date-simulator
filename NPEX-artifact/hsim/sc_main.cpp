#include <components/Top.h>

Configurations cfgs;
SyncObject sync_object;

int sc_main(int argc, char* argv[])
{
    cfgs.init_configurations();

    const double host_freq_ghz = cfgs.get_host_freq(); /* GHz */
    
    if (host_freq_ghz <= 0.0) {
        std::cerr << "[sc_main] Invalid host_freq (GHz): " << host_freq_ghz << "\n";
        return 1;
    }

    Top top("top");

    // Fast-layer macro timing is event driven.  A 10-us control clock is
    // sufficient to inject its ordered request stream while avoiding hundreds
    // of thousands of idle Host/Arbiter edges during long macro waits.
    const sc_core::sc_time host_period(
        cfgs.fast_layer_mode_enabled() ? 10000.0 : 1.0 / host_freq_ghz,
        sc_core::SC_NS);
    sc_core::sc_clock clk("clk", host_period);
    top.clock(clk);

    const auto wall_start = std::chrono::steady_clock::now();
    sc_core::sc_start();
    const auto wall_end = std::chrono::steady_clock::now();

    cfgs.print_configurations();

    const sc_core::sc_time sim_time = sc_core::sc_time_stamp();
    top.performance_metrics.write_json(cfgs.get_performance_stats_file(), sim_time);
    const std::uint64_t sim_cycles =
        static_cast<std::uint64_t>(sim_time / host_period);

    const std::chrono::duration<double> wall_sec = wall_end - wall_start;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Wall Time       : " << wall_sec.count() << " (seconds)\n";
    std::cout << "Simulated Time  : " << sim_time.to_seconds() * 1e9 << " (ns)\n";
    std::cout << "Simulated Cycles: " << sim_cycles << " (host cycles)\n";
    return 0;
}
