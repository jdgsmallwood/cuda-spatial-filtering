
add_executable(
  test_cublas
  ${PROJECT_SOURCE_DIR}/apps/test_cublas_complex.cpp
)

target_link_libraries(test_cublas PRIVATE cudart cuda cublas)

add_executable(
  test_cutensor
  ${PROJECT_SOURCE_DIR}/apps/test_cutensor.cpp
)

target_link_libraries(test_cutensor PRIVATE cudart cuda cutensor)

add_executable(
  test_cutensor_2
  ${PROJECT_SOURCE_DIR}/apps/test_cutensor_2.cpp
)

target_link_libraries(test_cutensor_2 PRIVATE cudart cuda cutensor spatial)

add_executable(
  beamform_spatial
  ${PROJECT_SOURCE_DIR}/apps/beamform_spatial.cpp
)
target_link_libraries(beamform_spatial PRIVATE cudart cuda spatial cutensor pcap HighFive::HighFive)

add_executable(
  read_pcap_and_beamform
  ${PROJECT_SOURCE_DIR}/apps/read_pcap_and_beamform.cpp
)
target_link_libraries(read_pcap_and_beamform PRIVATE spatial pcap cudart tcc cuda cutensor ccglib)
