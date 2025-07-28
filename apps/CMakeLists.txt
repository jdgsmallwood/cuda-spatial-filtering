
add_executable(
  test_cublas
  ${PROJECT_SOURCE_DIR}/apps/test_cublas_complex.cpp
)

target_link_libraries(test_cublas PRIVATE cuda cublas CUDA::cudart)

add_executable(
  test_cutensor
  ${PROJECT_SOURCE_DIR}/apps/test_cutensor.cpp
)

target_link_libraries(test_cutensor PRIVATE CUDA::cudart cuda cutensor)

add_executable(
  test_cutensor_2
  ${PROJECT_SOURCE_DIR}/apps/test_cutensor_2.cpp
)

target_link_libraries(test_cutensor_2 PRIVATE CUDA::cudart cuda cutensor spatial)

add_executable(
  beamform_spatial
  ${PROJECT_SOURCE_DIR}/apps/beamform_spatial.cpp
)
target_link_libraries(beamform_spatial PRIVATE CUDA::cudart cuda spatial cutensor pcap HighFive::HighFive ${HDF5_LIBRARIES} ${ZLIB_LIBRARIES})

message(STATUS "Benchmarking is $BENCHMARKING")

if(BENCHMARKING EQUAL 1)
  target_compile_definitions(beamform_spatial PRIVATE BENCHMARKING=1)
endif()

add_executable(
  read_pcap_and_beamform
  ${PROJECT_SOURCE_DIR}/apps/read_pcap_and_beamform.cpp
)
target_link_libraries(read_pcap_and_beamform PRIVATE spatial pcap CUDA::cudart tcc cuda cutensor ccglib)
