
add_executable(
    read_pcap
    ${PROJECT_SOURCE_DIR}/apps/read_pcap_file.cpp
)



target_link_libraries(read_pcap PRIVATE spatial pcap cudart tcc cuda)

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
  read_pcap_and_beamform
  ${PROJECT_SOURCE_DIR}/apps/read_pcap_and_beamform.cpp
)
target_link_libraries(read_pcap_and_beamform PRIVATE spatial pcap cudart tcc cuda cutensor)
