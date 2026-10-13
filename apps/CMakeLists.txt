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

foreach(NR_CHANNELS 56 64 72 80)
    add_executable(bench_starspin_pipeline_${NR_CHANNELS}_thp
        ${PROJECT_SOURCE_DIR}/apps/bench_starspin_pipeline.cu)
    target_link_libraries(bench_starspin_pipeline_${NR_CHANNELS}_thp PRIVATE gpu_app_common)
    target_compile_definitions(bench_starspin_pipeline_${NR_CHANNELS}_thp PRIVATE
        NR_OBSERVING_CHANNELS=${NR_CHANNELS}
        SPATIAL_THP_INPUT_BUFFERS=1
        SPATIAL_THP_OUTPUT_BUFFERS=1)
endforeach()

# Compare the 40-channel Starspin GPU path with 28 and 27 retained fine bins
# at the same integration cadence, independent of packet capture and HDF5.
foreach(NR_LOWER_EXTRA_TRIM 0 1)
    math(EXPR NR_RETAINED_FINE "28 - ${NR_LOWER_EXTRA_TRIM}")
    add_executable(bench_starspin_pipeline_40_${NR_RETAINED_FINE}
        ${PROJECT_SOURCE_DIR}/apps/bench_starspin_pipeline.cu)
    target_link_libraries(bench_starspin_pipeline_40_${NR_RETAINED_FINE}
        PRIVATE gpu_app_common)
    target_compile_definitions(bench_starspin_pipeline_40_${NR_RETAINED_FINE} PRIVATE
        NR_OBSERVING_CHANNELS=40
        NR_OBSERVING_FINE_CHANNEL_LOWER_EXTRA_TRIM=${NR_LOWER_EXTRA_TRIM}
        SPATIAL_THP_INPUT_BUFFERS=1
        SPATIAL_THP_OUTPUT_BUFFERS=1)
endforeach()

# Focused Starspin HDF5 writer throughput comparison.
add_executable(bench_starspin_vis_writer ${PROJECT_SOURCE_DIR}/apps/bench_starspin_vis_writer.cu)
target_link_libraries(bench_starspin_vis_writer PRIVATE gpu_app_common)

# Starspin grid: all FPGA source counts, using multiples of eight channels.
# The FPGA also supports other even counts from 8 through 330; explicit
# experimental overrides remain available via STARSPIN_CHANNEL_COUNTS.
# Keep the default set small so a normal build does not compile hundreds of
# CUDA specializations.
set(STARSPIN_CHANNEL_COUNTS "8;16;24;32;40;48;56;64;72;80" CACHE STRING
    "Even coarse channel counts (8..330) for Starspin targets")
foreach(NR_CHANNELS IN LISTS STARSPIN_CHANNEL_COUNTS)
    if(NOT NR_CHANNELS MATCHES "^[0-9]+$")
        message(FATAL_ERROR "Invalid Starspin channel count: ${NR_CHANNELS}")
    endif()
    math(EXPR _STARSPIN_CHANNEL_ODD "${NR_CHANNELS} % 2")
    if(NR_CHANNELS LESS 8 OR NR_CHANNELS GREATER 330 OR _STARSPIN_CHANNEL_ODD)
        message(FATAL_ERROR "Starspin channel counts must be even and between 8 and 330: ${NR_CHANNELS}")
    endif()
endforeach()
list(REMOVE_DUPLICATES STARSPIN_CHANNEL_COUNTS)
set(_STARSPIN_TARGETS)
foreach(NR_FPGAS 1 2 3 4)
    foreach(NR_CHANNELS IN LISTS STARSPIN_CHANNEL_COUNTS)
        set(SW_TARGET "starspin_${NR_FPGAS}_${NR_CHANNELS}")
        math(EXPR _NR_RECEIVERS "${NR_FPGAS} * ${NR_OBSERVING_RECEIVERS_PER_PACKET}")
        set(_ACTIVE_RECEIVERS ${_NR_RECEIVERS})
        if(NR_FPGAS EQUAL 4 AND NR_CHANNELS GREATER_EQUAL 40)
            # The canonical stream map places four disconnected receiver slots
            # last. TCC can correlate the 36 connected receivers directly.
            set(_ACTIVE_RECEIVERS 36)
        endif()
        add_executable(${SW_TARGET} ${PROJECT_SOURCE_DIR}/apps/starspin.cu)
        target_link_libraries(${SW_TARGET} PRIVATE gpu_app_common)
        # A RAW_PACKET ibverbs QP requires cap_net_raw.  Linux enters secure
        # loader mode for a file-capability executable and ignores
        # LD_LIBRARY_PATH, so embed the runtime locations Starspin needs.
        set_target_properties(${SW_TARGET} PROPERTIES BUILD_RPATH
            "${SPATIAL_APP_BUILD_RPATH}")
        target_compile_definitions(${SW_TARGET} PRIVATE
            NR_OBSERVING_FPGA_SOURCES=${NR_FPGAS}
            NR_OBSERVING_CHANNELS=${NR_CHANNELS}
            NR_OBSERVING_ACTIVE_RECEIVERS=${_ACTIVE_RECEIVERS}
            SPATIAL_THP_INPUT_BUFFERS=1
        )
        if(NR_FPGAS EQUAL 4 AND NR_CHANNELS GREATER_EQUAL 54)
            # The live 54-channel CPU ingest benchmark sustained the full
            # 3.125 Mpacket/s only with direct per-NIC workers and batched
            # completion watermark publication. The smaller NUMA-local rings
            # and native copy_nt ISA also improved the measured throughput.
            target_compile_definitions(${SW_TARGET} PRIVATE
                SPATIAL_PER_NIC_RING=1
                SPATIAL_PER_NIC_RING_SLOTS=1536
                SPATIAL_DIRECT_NIC_WORKERS=1
                SPATIAL_BATCH_PACKET_WATERMARKS=1)
            target_compile_options(${SW_TARGET} PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
        endif()
        if(NR_FPGAS EQUAL 4 AND NR_CHANNELS GREATER_EQUAL 48)
            target_compile_definitions(${SW_TARGET} PRIVATE
                SPATIAL_STARSPIN_WORKER_THREADS=4
                SPATIAL_PROCESSOR_BATCH_SIZE=256
            )
        endif()
        list(APPEND _STARSPIN_TARGETS ${SW_TARGET})
    endforeach()

    # Eight incoming FPGA channels, with no additional PFB channelization.
    set(COARSE_TARGET "starspin_${NR_FPGAS}_8_coarse")
    add_executable(${COARSE_TARGET} ${PROJECT_SOURCE_DIR}/apps/starspin.cu)
    target_link_libraries(${COARSE_TARGET} PRIVATE gpu_app_common)
    set_target_properties(${COARSE_TARGET} PROPERTIES BUILD_RPATH
        "${SPATIAL_APP_BUILD_RPATH}")
    target_compile_definitions(${COARSE_TARGET} PRIVATE
        NR_OBSERVING_FPGA_SOURCES=${NR_FPGAS}
        NR_OBSERVING_CHANNELS=8
        SPATIAL_STARSPIN_COARSE_ONLY=1
        SPATIAL_THP_INPUT_BUFFERS=1)
    list(APPEND _STARSPIN_TARGETS ${COARSE_TARGET})
endforeach()

# Retained end-to-end CPU ingest benchmark. It executes the normal packet
# capture and assembly path but releases completed host buffers immediately.
foreach(NR_CHANNELS IN LISTS STARSPIN_CHANNEL_COUNTS)
    set(CAPTURE_ONLY_TARGET "starspin_4_${NR_CHANNELS}_capture_only")
    add_executable(${CAPTURE_ONLY_TARGET} ${PROJECT_SOURCE_DIR}/apps/starspin.cu)
    target_link_libraries(${CAPTURE_ONLY_TARGET} PRIVATE gpu_app_common)
    set_target_properties(${CAPTURE_ONLY_TARGET} PROPERTIES BUILD_RPATH
        "${SPATIAL_APP_BUILD_RPATH}")
    target_compile_definitions(${CAPTURE_ONLY_TARGET} PRIVATE
        NR_OBSERVING_FPGA_SOURCES=4
        NR_OBSERVING_CHANNELS=${NR_CHANNELS}
        NR_OBSERVING_ACTIVE_RECEIVERS=36
        SPATIAL_CAPTURE_ONLY=1
        SPATIAL_THP_INPUT_BUFFERS=1)
    if(NR_CHANNELS GREATER_EQUAL 54)
        target_compile_definitions(${CAPTURE_ONLY_TARGET} PRIVATE
            SPATIAL_STARSPIN_WORKER_THREADS=4
            SPATIAL_PROCESSOR_BATCH_SIZE=256
            SPATIAL_PER_NIC_RING=1
            SPATIAL_PER_NIC_RING_SLOTS=1536
            SPATIAL_DIRECT_NIC_WORKERS=1
            SPATIAL_BATCH_PACKET_WATERMARKS=1)
        target_compile_options(${CAPTURE_ONLY_TARGET} PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)
    elseif(NR_CHANNELS GREATER_EQUAL 48)
        target_compile_definitions(${CAPTURE_ONLY_TARGET} PRIVATE
            SPATIAL_STARSPIN_WORKER_THREADS=4
            SPATIAL_PROCESSOR_BATCH_SIZE=256)
    endif()
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
add_executable(capture_delay_samples ${PROJECT_SOURCE_DIR}/apps/capture_delay_samples.cu)
add_executable(bench_starspin_ibverbs ${PROJECT_SOURCE_DIR}/apps/bench_starspin_ibverbs.cu)
target_link_libraries(bench_starspin_ibverbs PRIVATE gpu_app_common)
set_target_properties(bench_starspin_ibverbs PROPERTIES BUILD_RPATH "${SPATIAL_APP_BUILD_RPATH}")
add_library(delay_correlation SHARED ${PROJECT_SOURCE_DIR}/apps/delay_correlation.cu)
target_link_libraries(delay_correlation PRIVATE CUDA::cudart CUDA::cufft)
target_include_directories(capture_delay_samples PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(capture_delay_samples PRIVATE pcap argparse spdlog::spdlog tcc nlohmann_json::nlohmann_json)
set_target_properties(capture_delay_samples PROPERTIES BUILD_RPATH "${SPATIAL_APP_BUILD_RPATH}")
if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
    target_link_libraries(capture_delay_samples PRIVATE ${IBVERBS_LIBRARY})
endif()
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
# LD_LIBRARY_PATH. Embed the same runtime locations as the proven Starspin
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

add_executable(bench_starspin_h2d ${PROJECT_SOURCE_DIR}/apps/bench_starspin_h2d.cu)
target_link_libraries(bench_starspin_h2d PRIVATE CUDA::cudart)

foreach(NR_CHANNELS IN ITEMS 48 54)
    add_executable(bench_starspin_pipeline_${NR_CHANNELS}
        ${PROJECT_SOURCE_DIR}/apps/bench_starspin_pipeline.cu)
    target_link_libraries(bench_starspin_pipeline_${NR_CHANNELS} PRIVATE gpu_app_common)
    target_compile_definitions(bench_starspin_pipeline_${NR_CHANNELS} PRIVATE
        NR_OBSERVING_CHANNELS=${NR_CHANNELS})
endforeach()

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
foreach(NR_FPGAS 1 2 3 4)
    foreach(NR_CHANNELS 8 16 24 32 40 48)
        list(APPEND _OBSERVE_TARGETS "observe_${NR_FPGAS}_${NR_CHANNELS}")
    endforeach()
endforeach()

install(TARGETS
    ${_OBSERVE_TARGETS}
    ${_STARSPIN_TARGETS}
    ${_STANDARD_APPS}
    pulsar_fold_4_40
    udp_sender
    RUNTIME DESTINATION bin
)
