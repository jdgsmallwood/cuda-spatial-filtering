set(NUMBER_BEAMS 1 CACHE STRING "Number of beams to correlate while benchmarking")

add_library(gpu_app_common INTERFACE)

message(STATUS "PSRDADA_ROOT is ${PSRDADA_ROOT}")

target_include_directories(gpu_app_common INTERFACE 
    ${PSRDADA_ROOT}/include
)
target_compile_options(gpu_app_common INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128,2417>)
# target_compile_options(gpu_app_common INTERFACE
#     $<$<COMPILE_LANGUAGE:CUDA>:-w>)

# Opt-in: ties the host-code build to this machine's exact CPU ISA, so off by
# default (production apps stay portable across deployment hosts). Unlocks
# copy_nt's AVX-512 non-temporal-store path (spatial.hpp) -- the same code
# path bench_processor's own -march=native flag validated at 3.2->14.7 GB/s
# (see scripts/profiling/BENCH_PROCESSOR_TUNING.md). Without this flag,
# __AVX512F__ is never defined and copy_nt silently falls back to its SSE2
# tier even on hardware that supports AVX-512.
option(SPATIAL_NATIVE_ARCH "Enable -march=native for host code (ties the build to this CPU)" OFF)
option(SPATIAL_DIAGNOSTICS "Enable compile-time ingest diagnostics" OFF)
if(SPATIAL_NATIVE_ARCH)
    target_compile_options(gpu_app_common INTERFACE
        $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
endif()
if(SPATIAL_DIAGNOSTICS)
    target_compile_definitions(gpu_app_common INTERFACE SPATIAL_DIAGNOSTICS)
endif()
target_link_directories(gpu_app_common INTERFACE ${PSRDADA_ROOT}/lib)

# File capabilities (used by RAW_PACKET ibverbs) make the dynamic loader ignore
# LD_LIBRARY_PATH.  Derive normal runtime locations from this build; sites may
# supply any additional compatibility-library directories at configure time.
set(SPATIAL_EXTRA_RUNTIME_LIBRARY_DIRS "" CACHE STRING
    "Additional runtime library directories for capability-bearing executables")
set(SPATIAL_APP_BUILD_RPATH
    ${PSRDADA_ROOT}/lib
    ${CMAKE_BINARY_DIR}/_deps/mathdx-build
    ${CMAKE_BINARY_DIR}/_deps/ccglib-build/src
    ${CMAKE_BINARY_DIR}/_deps/tcc-build/libtcc
    ${SPATIAL_EXTRA_RUNTIME_LIBRARY_DIRS})
if(DEFINED ENV{CONDA_PREFIX})
    list(APPEND SPATIAL_APP_BUILD_RPATH
        "$ENV{CONDA_PREFIX}/lib"
        "$ENV{CONDA_PREFIX}/targets/x86_64-linux/lib")
endif()

target_link_libraries(gpu_app_common INTERFACE
    spatial CUDA::cudart cuda cutensor HighFive::HighFive
    spdlog::spdlog ccglib tcc filter ${CFITSIO_LIBRARIES} argparse
    ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft
    CUDA::cusolver nlohmann_json::nlohmann_json psrdada
)

target_compile_definitions(gpu_app_common INTERFACE
    NUMBER_BEAMS=${NUMBER_BEAMS}
    NR_OBSERVING_FINE_CHANNELS=${NR_OBSERVING_FINE_CHANNELS}
    NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM=${NR_OBSERVING_FINE_CHANNEL_EDGE_TRIM}
)

# Link libibverbs only when it was found at configure time (HAVE_IBVERBS),
# so the ibverbs capture backend resolves; no-op on non-RDMA machines.
if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
    target_link_libraries(gpu_app_common INTERFACE ${IBVERBS_LIBRARY})
endif()

# Observe grid: observe_{NR_FPGAS}_{NR_CHANNELS} for all combinations of
# FPGA sources (1-4) and channels (8,16,24,32,40,48).  PADDED_RECEIVERS is derived
# as the next multiple of 32 above NR_FPGAS * NR_OBSERVING_RECEIVERS_PER_PACKET.
foreach(NR_FPGAS 1 2 3 4)
    foreach(NR_CHANNELS 8 16 24 32 40 48)
        set(OBS_TARGET "observe_${NR_FPGAS}_${NR_CHANNELS}")
        math(EXPR _NR_RECEIVERS "${NR_FPGAS} * ${NR_OBSERVING_RECEIVERS_PER_PACKET}")
        math(EXPR _PADDED_RECEIVERS "(${_NR_RECEIVERS} + 31) / 32 * 32")
        add_executable(${OBS_TARGET} ${PROJECT_SOURCE_DIR}/apps/observe.cu)
        target_link_libraries(${OBS_TARGET} PRIVATE gpu_app_common)
        target_compile_definitions(${OBS_TARGET} PRIVATE
            NR_OBSERVING_FPGA_SOURCES=${NR_FPGAS}
            NR_OBSERVING_CHANNELS=${NR_CHANNELS}
            NR_OBSERVING_PADDED_RECEIVERS=${_PADDED_RECEIVERS}
        )
    endforeach()
endforeach()

# Starweave grid: starweave_{NR_FPGAS}_{NR_CHANNELS}, same combinations as the observe grid
# above -- correlation-only, visibilities-only counterpart to observe (see
# apps/starweave.cu / LambdaStarweavePipeline).
foreach(NR_FPGAS 1 2 3 4)
    foreach(NR_CHANNELS 8 16 24 32 40 48)
        set(SW_TARGET "starweave_${NR_FPGAS}_${NR_CHANNELS}")
        math(EXPR _NR_RECEIVERS "${NR_FPGAS} * ${NR_OBSERVING_RECEIVERS_PER_PACKET}")
        math(EXPR _PADDED_RECEIVERS "(${_NR_RECEIVERS} + 31) / 32 * 32")
        add_executable(${SW_TARGET} ${PROJECT_SOURCE_DIR}/apps/starweave.cu)
        target_link_libraries(${SW_TARGET} PRIVATE gpu_app_common)
        # A RAW_PACKET ibverbs QP requires cap_net_raw.  Linux enters secure
        # loader mode for a file-capability executable and ignores
        # LD_LIBRARY_PATH, so embed the runtime locations Starweave needs.
        set_target_properties(${SW_TARGET} PROPERTIES BUILD_RPATH
            "${SPATIAL_APP_BUILD_RPATH}")
        target_compile_definitions(${SW_TARGET} PRIVATE
            NR_OBSERVING_FPGA_SOURCES=${NR_FPGAS}
            NR_OBSERVING_CHANNELS=${NR_CHANNELS}
            NR_OBSERVING_PADDED_RECEIVERS=${_PADDED_RECEIVERS}
        )
    endforeach()
endforeach()

add_executable(udp_sender ${PROJECT_SOURCE_DIR}/apps/udp_sender.cpp)
target_link_libraries(udp_sender PRIVATE pcap spdlog::spdlog)

# Raw-frame pcap replay over a libibverbs send QP, for testing
# LibibverbsPacketCapture/LibibverbsGpuDirectPacketCapture end to end (send
# via ibverbs, receive via ibverbs) -- see the file's header comment. Prints
# a friendly error and exits if built without libibverbs, matching the
# capture-side backend's own --capture-backend=ibverbs error message.
add_executable(ibverbs_pcap_sender ${PROJECT_SOURCE_DIR}/apps/ibverbs_pcap_sender.cpp)
# The ibverbs sender includes the GPUDirect relocation kernel and its CUDA
# launch syntax through libibverbs.hpp, so this .cpp source must be compiled
# by nvcc rather than the host C++ compiler.
set_source_files_properties(${PROJECT_SOURCE_DIR}/apps/ibverbs_pcap_sender.cpp
    PROPERTIES LANGUAGE CUDA)
target_include_directories(ibverbs_pcap_sender PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ibverbs_pcap_sender PRIVATE pcap argparse spdlog::spdlog tcc)
if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
    target_link_libraries(ibverbs_pcap_sender PRIVATE ${IBVERBS_LIBRARY})
endif()

add_executable(gpu_benchmark ${PROJECT_SOURCE_DIR}/apps/gpu_benchmark.cu)
target_link_libraries(gpu_benchmark PRIVATE gpu_app_common)

add_executable(fft_antenna_spectra ${PROJECT_SOURCE_DIR}/apps/fft_antenna_spectra.cu)
target_link_libraries(fft_antenna_spectra PRIVATE gpu_app_common)

add_executable(beamformed_bandpass ${PROJECT_SOURCE_DIR}/apps/beamformed-bandpass.cu)
target_link_libraries(beamformed_bandpass PRIVATE gpu_app_common)

add_executable(get_projection_matrix ${PROJECT_SOURCE_DIR}/apps/get_projection_matrix.cu)
target_link_libraries(get_projection_matrix PRIVATE gpu_app_common)

add_executable(adaptive_beamformed_bandpass ${PROJECT_SOURCE_DIR}/apps/adaptive-beamformed-bandpass.cu)
target_link_libraries(adaptive_beamformed_bandpass PRIVATE gpu_app_common)

add_executable(pulsar_fold ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
target_link_libraries(pulsar_fold PRIVATE gpu_app_common)

add_executable(pointing_enu ${PROJECT_SOURCE_DIR}/apps/pointing_enu.cpp
    $<TARGET_OBJECTS:pointing>)
target_include_directories(pointing_enu PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(pointing_enu PRIVATE ${CASACORE_LIBRARIES})

# Commissioning/science pulsar target: the known-good four-FPGA, 40-coarse-
# channel ingest shape, followed by the configured 32-way fine PFB (28 bins
# retained per coarse channel).  Keep this separate from the cache-shaped
# pulsar_fold target so a routine CMake reconfigure cannot silently build an
# 8-channel/one-FPGA binary for a 4x40 observation.
add_executable(pulsar_fold_4_40 ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
target_link_libraries(pulsar_fold_4_40 PRIVATE gpu_app_common)
target_compile_definitions(pulsar_fold_4_40 PRIVATE
    NR_OBSERVING_FPGA_SOURCES=4
    NR_OBSERVING_CHANNELS=40
    NR_OBSERVING_PADDED_RECEIVERS=64
)
# cap_net_raw puts the executable in secure-loader mode, which ignores
# LD_LIBRARY_PATH. Embed the same runtime locations as the proven Starweave
# ibverbs binaries.
set_target_properties(pulsar_fold_4_40 PROPERTIES BUILD_RPATH
    "${SPATIAL_APP_BUILD_RPATH}")

# Voltage-output control target matching the 24-coarse-channel backend mode.
# This provides an independent check of the detected-intensity/DSPSR path.
add_executable(pulsar_fold_4_24 ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
target_link_libraries(pulsar_fold_4_24 PRIVATE gpu_app_common)
target_compile_definitions(pulsar_fold_4_24 PRIVATE
    NR_OBSERVING_FPGA_SOURCES=4
    NR_OBSERVING_CHANNELS=24
    NR_OBSERVING_PADDED_RECEIVERS=64
)
set_target_properties(pulsar_fold_4_24 PROPERTIES BUILD_RPATH
    "${SPATIAL_APP_BUILD_RPATH}")

# Detected Stokes-I variant for folding on a single GPU.  Calibration and
# coherent beamforming still operate on complex X/Y fine-channel voltages;
# only the final DADA product is reduced to |X|^2 + |Y|^2.
add_executable(pulsar_fold_intensity_4_40 ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
target_link_libraries(pulsar_fold_intensity_4_40 PRIVATE gpu_app_common)
target_compile_definitions(pulsar_fold_intensity_4_40 PRIVATE
    NR_OBSERVING_FPGA_SOURCES=4
    NR_OBSERVING_CHANNELS=40
    NR_OBSERVING_PADDED_RECEIVERS=64
    PULSAR_DADA_DETECTED_INTENSITY=1
)
set_target_properties(pulsar_fold_intensity_4_40 PROPERTIES BUILD_RPATH
    "${SPATIAL_APP_BUILD_RPATH}")

# Smaller control point for the single-GPU pulsar path.  This uses the same
# four FPGA ingest, fine channelizer, calibration and detected-intensity output
# as the 40-channel target, but only channels [min, min+24).
add_executable(pulsar_fold_intensity_4_24 ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
target_link_libraries(pulsar_fold_intensity_4_24 PRIVATE gpu_app_common)
target_compile_definitions(pulsar_fold_intensity_4_24 PRIVATE
    NR_OBSERVING_FPGA_SOURCES=4
    NR_OBSERVING_CHANNELS=24
    NR_OBSERVING_PADDED_RECEIVERS=64
    PULSAR_DADA_DETECTED_INTENSITY=1
)
set_target_properties(pulsar_fold_intensity_4_24 PROPERTIES BUILD_RPATH
    "${SPATIAL_APP_BUILD_RPATH}")

# Compact four-FPGA replay targets: one coarse channel keeps a >10-second
# synthetic PCAP small enough for routine end-to-end DSPSR validation.
foreach(_mode voltage intensity)
    set(_target "pulsar_fold_replay_${_mode}_4_1")
    add_executable(${_target} ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
    target_link_libraries(${_target} PRIVATE gpu_app_common)
    target_compile_definitions(${_target} PRIVATE
        NR_OBSERVING_FPGA_SOURCES=4
        NR_OBSERVING_CHANNELS=1
        NR_OBSERVING_PADDED_RECEIVERS=64
    )
    if(_mode STREQUAL "intensity")
        target_compile_definitions(${_target} PRIVATE PULSAR_DADA_DETECTED_INTENSITY=1)
    endif()
    set_target_properties(${_target} PROPERTIES BUILD_RPATH
        "${SPATIAL_APP_BUILD_RPATH}")
endforeach()

add_executable(pulsar_fold_intensity_4_32 ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu)
target_link_libraries(pulsar_fold_intensity_4_32 PRIVATE gpu_app_common)
target_compile_definitions(pulsar_fold_intensity_4_32 PRIVATE
    NR_OBSERVING_FPGA_SOURCES=4
    NR_OBSERVING_CHANNELS=32
    NR_OBSERVING_PADDED_RECEIVERS=64
    PULSAR_DADA_DETECTED_INTENSITY=1
)
set_target_properties(pulsar_fold_intensity_4_32 PROPERTIES BUILD_RPATH
    "${SPATIAL_APP_BUILD_RPATH}")

# Microbenchmark binaries.
# bench_processor feeds synthetic packets, so it needs the tests/support headers.
add_executable(bench_capture ${PROJECT_SOURCE_DIR}/apps/bench_capture.cu)
target_link_libraries(bench_capture PRIVATE gpu_app_common)

add_executable(bench_processor ${PROJECT_SOURCE_DIR}/apps/bench_processor.cu)
target_include_directories(bench_processor PRIVATE ${PROJECT_SOURCE_DIR}/tests/support)
target_link_libraries(bench_processor PRIVATE gpu_app_common)
# Enable full native CPU ISA for the bench host code: unlocks AVX-512 in copy_nt
# (40 NT stores per 2560-byte copy vs 160 SSE2), better autovectorisation, and
# Zen 4 / Ice Lake scheduling.  Bench-only; production apps stay portable.
target_compile_options(bench_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)

add_executable(bench_writers ${PROJECT_SOURCE_DIR}/apps/bench_writers.cu)
target_link_libraries(bench_writers PRIVATE gpu_app_common)

add_executable(bench_gpu ${PROJECT_SOURCE_DIR}/apps/bench_gpu.cu)
target_link_libraries(bench_gpu PRIVATE gpu_app_common)

# Exploratory: eigendecomposition speedup investigation, not wired into the
# production pipeline -- see exploration/eigensolver/README.md for full
# context (findings, open questions, how to resume). Kept buildable here so
# it doesn't bit-rot.
add_executable(bench_eigensolver ${PROJECT_SOURCE_DIR}/exploration/eigensolver/bench_eigensolver.cu)
target_include_directories(bench_eigensolver PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                     ${PROJECT_SOURCE_DIR}/exploration/eigensolver)
target_link_libraries(bench_eigensolver PRIVATE CUDA::cudart CUDA::cusolver CUDA::cublas argparse spdlog::spdlog)

# Phase 0 validation benchmark for the GPUDirect RDMA ingest plan (see
# docs/architecture.md and the plan referenced there): CUDA-graph-replay
# scatter relocation throughput/contention, no ibverbs/NIC dependency.
add_executable(bench_gpudirect_scatter ${PROJECT_SOURCE_DIR}/apps/bench_gpudirect_scatter.cu)
target_include_directories(bench_gpudirect_scatter PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(bench_gpudirect_scatter PRIVATE spdlog::spdlog argparse CUDA::cudart cuda)

# Non-observe apps get the default (CMake-cache) values for the three macros
# that were removed from the global add_compile_definitions in favour of
# per-target definitions in the observe grid above.
set(_STANDARD_APPS
    gpu_benchmark fft_antenna_spectra beamformed_bandpass
    get_projection_matrix adaptive_beamformed_bandpass
    pulsar_fold bench_capture bench_processor bench_writers bench_gpu
)
foreach(_APP IN LISTS _STANDARD_APPS)
    target_compile_definitions(${_APP} PRIVATE
        NR_OBSERVING_FPGA_SOURCES=${NR_OBSERVING_FPGA_SOURCES}
        NR_OBSERVING_CHANNELS=${NR_OBSERVING_CHANNELS}
        NR_OBSERVING_PADDED_RECEIVERS=${NR_OBSERVING_PADDED_RECEIVERS}
    )
endforeach()

# Install all app binaries to <install-prefix>/bin (defaults to <project-root>/bin).
set(_OBSERVE_TARGETS)
set(_STARWEAVE_TARGETS)
foreach(NR_FPGAS 1 2 3 4)
    foreach(NR_CHANNELS 8 16 24 32 40 48)
        list(APPEND _OBSERVE_TARGETS "observe_${NR_FPGAS}_${NR_CHANNELS}")
        list(APPEND _STARWEAVE_TARGETS "starweave_${NR_FPGAS}_${NR_CHANNELS}")
    endforeach()
endforeach()

install(TARGETS
    ${_OBSERVE_TARGETS}
    ${_STARWEAVE_TARGETS}
    ${_STANDARD_APPS}
    pulsar_fold_4_40
    udp_sender
    RUNTIME DESTINATION bin
)
