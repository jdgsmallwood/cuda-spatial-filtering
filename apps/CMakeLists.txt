set(NUMBER_BEAMS 10 CACHE STRING "Number of beams to correlate while benchmarking")
add_executable(
  observe
  ${PROJECT_SOURCE_DIR}/apps/benchmark-20251030.cu
)
target_link_libraries(observe PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver)
target_compile_definitions(observe PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

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
target_link_libraries(gpu_benchmark PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver)

set_target_properties(gpu_benchmark PROPERTIES CUDA_ARCHITECTURES "native")
target_compile_definitions(gpu_benchmark PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

add_executable(
  fft_antenna_spectra
  ${PROJECT_SOURCE_DIR}/apps/fft_antenna_spectra.cu
)
target_link_libraries(fft_antenna_spectra PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver)
target_compile_definitions(fft_antenna_spectra PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

set_target_properties(fft_antenna_spectra PROPERTIES CUDA_ARCHITECTURES "native")

add_executable(
  beamformed_bandpass
  ${PROJECT_SOURCE_DIR}/apps/beamformed-bandpass.cu
)
target_link_libraries(beamformed_bandpass PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis++ hiredis CUDA::cufft CUDA::cusolver)
target_compile_definitions(beamformed_bandpass PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS})

set_target_properties(beamformed_bandpass PROPERTIES CUDA_ARCHITECTURES "native")
