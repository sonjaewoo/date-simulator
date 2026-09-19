cmake_minimum_required(VERSION 3.14)
project(hsim)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(.
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/systemc-3.0.0/include
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/jsoncpp/include
	${CMAKE_SOURCE_DIR}/src
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/src
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/ext/headers
	)

link_directories(
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/systemc-3.0.0/lib-linux64
	${CMAKE_CURRENT_SOURCE_DIR}/../libs/jsoncpp
	${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3
	)

add_definitions(
	-DROOT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../"
	-DDRAM_DRAMSIM3
	-DDRAMSIM3_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../DRAMsim3/"
	)

link_libraries(rt systemc-3.0.0 jsoncpp dramsim3)

add_executable(hsim
	sc_main.cpp
	components/host.cpp
	components/arbiter.cpp
	components/mem_wrapper.cpp
	components/pim_wrapper.cpp
	components/dram_macro_model.cpp
	components/trace_generator.cpp
	components/npu_trace_replayer.cpp
	utilities/mm.cpp
	utilities/logger.cpp
	utilities/performance_metrics.cpp
	utilities/configurations.cpp
	scenarios/speculative/s1.cpp
	)

# Standalone DRAMsim3 saturation benchmark.  This deliberately stays outside
# the SystemC simulator topology so channel calibration can be inspected
# quickly and independently of a speculative-decoding trace.
add_executable(dramsim3_channel_sweep
	tools/dramsim3_channel_sweep.cpp
	)
