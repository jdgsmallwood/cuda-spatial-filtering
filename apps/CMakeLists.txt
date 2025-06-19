
add_executable(
    read_pcap
    ${PROJECT_SOURCE_DIR}/apps/read_pcap_file.cpp
)

target_link_libraries(read_pcap PRIVATE spatial pcap cudart tcc cuda)
