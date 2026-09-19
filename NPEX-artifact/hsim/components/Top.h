#ifndef TOP_H
#define TOP_H

#include "host.h"
#include "npu_trace_replayer.h"
#include "mem_wrapper.h"
#include "pim_wrapper.h"
#include "arbiter.h"

unsigned int active_host;
unsigned int active_core;
unsigned int active_dram;
extern Configurations cfgs;

SyncObject dram_sync_obj;
SyncObject pim_sync_obj;
SyncObject pim_npu_sync_obj;

SC_MODULE(Top)
{
    std::unique_ptr<Host>         host;
    std::unique_ptr<Arbiter>      arbiter;
    std::unique_ptr<NPUTraceReplayer> synthetic_npu;
    std::unique_ptr<MEMWrapper>   dram;
    std::unique_ptr<PIMWrapper>   pim;
    sc_in<bool> clock;
    Logger logger;
    PerformanceMetrics performance_metrics;

    SC_CTOR(Top)
    {
        instantiate_modules();
        bind_modules();
        bind_clocks();

        active_host = 1;
        active_core = 1;
        active_dram = 1;
    }

private:
    void instantiate_modules() {
        host    = std::make_unique<Host>("host", &logger, &performance_metrics);
        synthetic_npu = std::make_unique<NPUTraceReplayer>("synthetic_npu", &logger,
                                                            &performance_metrics);
        arbiter = std::make_unique<Arbiter>("arbiter");

        if (cfgs.dram_enabled()) {
            dram = std::make_unique<MEMWrapper>("mem_wrapper", &logger, &performance_metrics);
        }
        if (cfgs.pim_enabled()) {
            pim = std::make_unique<PIMWrapper>("pim", &logger, &performance_metrics);
        }
    }

    void bind_modules() {
        host->master.bind(synthetic_npu->slave);
        host->master.bind(arbiter->slave);
        // Arbiter assumes its target-socket index 0 is Host and index 1 is
        // NPU.  Preserve that order for both the real MIDAP and trace replay.
        synthetic_npu->master.bind(arbiter->slave);

        if (dram) { arbiter->master.bind(dram->slave); }
        if (pim)  { arbiter->master.bind(pim->slave);  }
    }

    void bind_clocks() {
        host->clock(clock);
        arbiter->clock(clock);
        synthetic_npu->clock(clock);

        if (dram) { dram->clock(clock);    }
        if (pim)  { pim->clock(clock);     }
    }
};

#endif //TOP_H
