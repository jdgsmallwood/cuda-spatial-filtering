set(NUMBER_BEAMS 1 CACHE STRING "Number of beams to correlate while benchmarking")

set(CMAKE_CUDA_ARCHITECTURES "native")

add_library(gpu_app_common INTERFACE)

message(STATUS "PSRDADA_ROOT is ${PSRDADA_ROOT}")

target_include_directories(gpu_app_common INTERFACE 
    ${PSRDADA_ROOT}/include
)

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

