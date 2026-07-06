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

add_executable(observe ${PROJECT_SOURCE_DIR}/apps/observe.cu)
target_link_libraries(observe PRIVATE gpu_app_common)

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
set(BENCH_PROCESSOR_RING_BUFFER_SIZE "8192" CACHE STRING
    "Ring slots used by apps/bench_processor")
set(BENCH_PROCESSOR_WORKER_COUNT "6" CACHE STRING
    "Processor worker threads used by apps/bench_processor")
set(SPATIAL_PROCESSOR_REGULAR_BATCH_SIZE "6000" CACHE STRING
    "Packet batch size dispatched by ProcessorState::process_packets")
target_compile_definitions(bench_processor PRIVATE
    BENCH_PROCESSOR_RING_BUFFER_SIZE=${BENCH_PROCESSOR_RING_BUFFER_SIZE}
    BENCH_PROCESSOR_WORKER_COUNT=${BENCH_PROCESSOR_WORKER_COUNT}
    SPATIAL_PROCESSOR_REGULAR_BATCH_SIZE=${SPATIAL_PROCESSOR_REGULAR_BATCH_SIZE})
# Enable full native CPU ISA for the bench host code: unlocks AVX-512 in copy_nt
# (40 NT stores per 2560-byte copy vs 160 SSE2), better autovectorisation, and
# Zen 4 / Ice Lake scheduling.  Bench-only; production apps stay portable.
target_compile_options(bench_processor PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-march=native>)

add_executable(bench_writers ${PROJECT_SOURCE_DIR}/apps/bench_writers.cu)
target_link_libraries(bench_writers PRIVATE gpu_app_common)

add_executable(bench_gpu ${PROJECT_SOURCE_DIR}/apps/bench_gpu.cu)
target_link_libraries(bench_gpu PRIVATE gpu_app_common)
