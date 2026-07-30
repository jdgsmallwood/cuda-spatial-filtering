set(NUMBER_BEAMS 1 CACHE STRING "Number of beams to correlate while benchmarking")

set(CMAKE_CUDA_ARCHITECTURES "native")

add_library(gpu_app_common INTERFACE)

message(STATUS "PSRDADA_ROOT is ${PSRDADA_ROOT}")

target_include_directories(gpu_app_common INTERFACE 
    ${PSRDADA_ROOT}/include
)
target_compile_options(gpu_app_common INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128,2417>)
# target_compile_options(gpu_app_common INTERFACE
#     $<$<COMPILE_LANGUAGE:CUDA>:-w>)
target_link_directories(gpu_app_common INTERFACE ${PSRDADA_ROOT}/lib)

target_link_libraries(gpu_app_common INTERFACE 
    spatial CUDA::cudart cuda cutensor HighFive::HighFive 
    spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse 
    ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft 
    CUDA::cusolver nlohmann_json::nlohmann_json psrdada
)

target_compile_definitions(gpu_app_common INTERFACE 
    NUMBER_BEAMS=${NUMBER_BEAMS}
)

# Link libibverbs only when it was found at configure time (HAVE_IBVERBS),
# so the ibverbs capture backend resolves; no-op on non-RDMA machines.
if(IBVERBS_LIBRARY AND IBVERBS_INCLUDE_DIR)
    target_link_libraries(gpu_app_common INTERFACE ${IBVERBS_LIBRARY})
endif()

# Observe grid: observe_{NR_FPGAS}_{NR_CHANNELS} for all combinations of
# FPGA sources (1-4) and channels (8,16,24,32).  PADDED_RECEIVERS is derived
# as the next multiple of 32 above NR_FPGAS * NR_OBSERVING_RECEIVERS_PER_PACKET.
foreach(NR_FPGAS 1 2 3 4)
    foreach(NR_CHANNELS 8 16 24 32)
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

add_executable(udp_sender ${PROJECT_SOURCE_DIR}/apps/udp_sender.cpp)
target_link_libraries(udp_sender PRIVATE pcap spdlog::spdlog)

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
    foreach(NR_CHANNELS 8 16 24 32)
        list(APPEND _OBSERVE_TARGETS "observe_${NR_FPGAS}_${NR_CHANNELS}")
    endforeach()
endforeach()

install(TARGETS
    ${_OBSERVE_TARGETS}
    ${_STANDARD_APPS}
    udp_sender
    RUNTIME DESTINATION bin
)

