set(NUMBER_BEAMS 1 CACHE STRING "Number of beams to correlate while benchmarking")
add_executable(
  observe
  ${PROJECT_SOURCE_DIR}/apps/observe.cu
)
target_link_libraries(observe PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)
target_compile_definitions(observe PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

target_include_directories(observe PRIVATE
    ${PSRDADA_ROOT}/include
)
#
set_target_properties(observe PROPERTIES CUDA_ARCHITECTURES "native")

add_executable(
  udp_sender
  ${PROJECT_SOURCE_DIR}/apps/udp_sender.cpp
)
target_link_libraries(udp_sender PRIVATE pcap spdlog::spdlog)


add_executable(
  gpu_benchmark
  ${PROJECT_SOURCE_DIR}/apps/gpu_benchmark.cu
)
target_link_libraries(gpu_benchmark PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)

target_include_directories(gpu_benchmark PRIVATE
    ${PSRDADA_ROOT}/include
)
#
set_target_properties(gpu_benchmark PROPERTIES CUDA_ARCHITECTURES "native")
target_compile_definitions(gpu_benchmark PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

add_executable(
  fft_antenna_spectra
  ${PROJECT_SOURCE_DIR}/apps/fft_antenna_spectra.cu
)
target_link_libraries(fft_antenna_spectra PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)
target_compile_definitions(fft_antenna_spectra PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

target_include_directories(fft_antenna_spectra PRIVATE
    ${PSRDADA_ROOT}/include
)
#
set_target_properties(fft_antenna_spectra PROPERTIES CUDA_ARCHITECTURES "native")

add_executable(
  beamformed_bandpass
  ${PROJECT_SOURCE_DIR}/apps/beamformed-bandpass.cu
)
target_link_libraries(beamformed_bandpass PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)
target_compile_definitions(beamformed_bandpass PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

target_include_directories(beamformed_bandpass PRIVATE
    ${PSRDADA_ROOT}/include
)
#
set_target_properties(beamformed_bandpass PROPERTIES CUDA_ARCHITECTURES "native")

add_executable(
  get_projection_matrix
  ${PROJECT_SOURCE_DIR}/apps/get_projection_matrix.cu
)
target_link_libraries(get_projection_matrix PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)
set_target_properties(get_projection_matrix PROPERTIES CUDA_ARCHITECTURES "native")

target_include_directories(get_projection_matrix PRIVATE
    ${PSRDADA_ROOT}/include
)
#
add_executable(
  adaptive_beamformed_bandpass
  ${PROJECT_SOURCE_DIR}/apps/adaptive-beamformed-bandpass.cu
)
target_link_libraries(adaptive_beamformed_bandpass PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)
target_compile_definitions(adaptive_beamformed_bandpass PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

target_include_directories(adaptive_beamformed_bandpass PRIVATE
    ${PSRDADA_ROOT}/include
)
#
set_target_properties(adaptive_beamformed_bandpass PROPERTIES CUDA_ARCHITECTURES "native")


add_executable(
  pulsar_fold
  ${PROJECT_SOURCE_DIR}/apps/pulsar-fold.cu
)

target_link_libraries(pulsar_fold PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver nlohmann_json::nlohmann_json psrdada)
target_compile_definitions(pulsar_fold PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})


set_target_properties(pulsar_fold PROPERTIES CUDA_ARCHITECTURES "native")
# for pulsar work later.
target_include_directories(pulsar_fold PRIVATE
    ${PSRDADA_ROOT}/include
)
#
#
#target_link_libraries(myprog PRIVATE
#    psrdada
#)
