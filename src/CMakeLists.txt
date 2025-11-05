

add_library(spatial 
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cpp
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
${CMAKE_CURRENT_SOURCE_DIR}/ethernet.cpp
${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
${CMAKE_CURRENT_SOURCE_DIR}/packet_formats.cpp
${CMAKE_CURRENT_SOURCE_DIR}/pipeline.cpp
)


target_link_libraries(
    spatial PRIVATE cusolver cuda cudart cublas tcc CUDA::cudart ccglib pcap cutensor HighFive::HighFive spdlog::spdlog
)


target_include_directories(
    spatial
    PUBLIC
        ${PROJECT_SOURCE_DIR}/extern/tcc
        ${PROJECT_SOURCE_DIR}/extern/libpcap
        ${PROJECT_SOURCE_DIR}/extern/cudawrappers/include
        ${PROJECT_SOURCE_DIR}/extern/spdlog/include
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(spatial PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


if(CMAKE_BUILD_TYPE MATCHES Debug)
  #target_link_libraries(spatial PRIVATE sanitizer_flags)
endif()
