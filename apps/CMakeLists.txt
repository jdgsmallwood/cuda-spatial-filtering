
#add_executable(
#  test_cublas
#  ${PROJECT_SOURCE_DIR}/apps/test_cublas_complex.cpp
#)
#
#target_link_libraries(test_cublas PRIVATE cuda cublas CUDA::cudart)
#
#add_executable(
#  test_cutensor
#  ${PROJECT_SOURCE_DIR}/apps/test_cutensor.cpp
#)
#
#target_link_libraries(test_cutensor PRIVATE CUDA::cudart cuda cutensor)
#
#add_executable(
#  test_cutensor_2
#  ${PROJECT_SOURCE_DIR}/apps/test_cutensor_2.cpp
#)
#
#target_link_libraries(test_cutensor_2 PRIVATE CUDA::cudart cuda cutensor spatial)

#add_executable(
#  beamform_spatial
#  ${PROJECT_SOURCE_DIR}/apps/beamform_spatial.cpp
#)
#target_link_libraries(beamform_spatial PRIVATE CUDA::cudart cuda spatial cutensor pcap HighFive::HighFive ${HDF5_LIBRARIES} ${ZLIB_LIBRARIES})

#message(STATUS "Benchmarking is $BENCHMARKING")
#
#if(BENCHMARKING EQUAL 1)
#  target_compile_definitions(beamform_spatial PRIVATE BENCHMARKING=1)
#endif()

#add_executable(
#  read_pcap_and_beamform
#  ${PROJECT_SOURCE_DIR}/apps/read_pcap_and_beamform.cpp
#)
#target_link_libraries(read_pcap_and_beamform PRIVATE spatial pcap CUDA::cudart tcc cuda cutensor ccglib)


#add_executable(
#  packet_processor
#  ${PROJECT_SOURCE_DIR}/apps/packet_processor.cpp
#)
#target_link_libraries(packet_processor PRIVATE spatial CUDA::cudart cuda cutensor)

#add_executable(
#  packet_processor_live
#  ${PROJECT_SOURCE_DIR}/apps/packet_processor_live.cu
#)
#target_link_libraries(packet_processor_live PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc)
set(NUMBER_PACKETS_TO_CORRELATE 16 CACHE STRING "Number of packets to correlate while benchmarking")
set(NUMBER_BEAMS 10 CACHE STRING "Number of beams to correlate while benchmarking")
add_executable(
  benchmark
  ${PROJECT_SOURCE_DIR}/apps/benchmark-20251030.cu
)
target_link_libraries(benchmark PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES} redis_plus_plus hiredis)
target_compile_definitions(benchmark PRIVATE NUMBER_BEAMS=${NUMBER_BEAMS} NUMBER_PACKETS_TO_CORRELATE=${NUMBER_PACKETS_TO_CORRELATE} )

set_target_properties(benchmark PROPERTIES CUDA_ARCHITECTURES "native")

add_executable(
  udp_sender
  ${PROJECT_SOURCE_DIR}/apps/udp_sender.cpp
)
target_link_libraries(udp_sender PRIVATE pcap spdlog::spdlog)


add_executable(
  gpu_benchmark
  ${PROJECT_SOURCE_DIR}/apps/gpu_benchmark.cu
)
target_link_libraries(gpu_benchmark PRIVATE spatial CUDA::cudart cuda cutensor  HighFive::HighFive spdlog::spdlog ccglib tcc ${CFITSIO_LIBRARIES} argparse ${CASACORE_LIBRARIES})

set_target_properties(gpu_benchmark PROPERTIES CUDA_ARCHITECTURES "native")
