#ifndef COMMON_H
#define COMMON_H

#include <tuple>
#include <string>
#include <unordered_map>

/* (layer, (id, size, address)) */
using BaseMap = std::unordered_multimap<std::string, std::tuple<unsigned int, unsigned int, uint64_t>>;

using SyncMap = BaseMap;
using WaitMap = BaseMap;

enum DeviceType {
    HOST = 0,
    NPU,
    PIM,
    DRAM,
    MCU,
    NONE
};

#endif