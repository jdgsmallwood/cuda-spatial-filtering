

add_library(spatial 
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cpp
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
${CMAKE_CURRENT_SOURCE_DIR}/ethernet.cpp
${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
)


target_link_libraries(
    spatial PRIVATE cusolver cuda cudart cublas tcc CUDA::cudart ccglib pcap cutensor
)


target_include_directories(
    spatial
    PUBLIC
        ${PROJECT_SOURCE_DIR}/extern/tcc
        ${PROJECT_SOURCE_DIR}/extern/libpcap
        ${PROJECT_SOURCE_DIR}/extern/cudawrappers/include
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(spatial PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

message(STATUS "Src/ Test building is ${USER_BUILD_TESTING}")
if(USER_BUILD_TESTING) 
  message(STATUS "building 8bit & 16bit versions of spatial...") 
add_library(spatial_8bit 
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cpp
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
${CMAKE_CURRENT_SOURCE_DIR}/ethernet.cpp
${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
)


target_link_libraries(
    spatial_8bit PRIVATE cusolver cuda cudart cublas tcc CUDA::cudart ccglib pcap cutensor
)
  target_compile_definitions(spatial_8bit PRIVATE NR_RECEIVERS=32 NR_BITS=8 NR_RECEIVERS_PER_BLOCK=32)

target_include_directories(
    spatial_8bit
    PUBLIC
        ${PROJECT_SOURCE_DIR}/extern/tcc
        ${PROJECT_SOURCE_DIR}/extern/libpcap
        ${PROJECT_SOURCE_DIR}/extern/cudawrappers/include
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(spatial_8bit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)


  
add_library(spatial_16bit 
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cpp
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
${CMAKE_CURRENT_SOURCE_DIR}/ethernet.cpp
${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
)


target_link_libraries(
    spatial_16bit PRIVATE cusolver cuda cudart cublas tcc CUDA::cudart ccglib pcap cutensor
)
  target_compile_definitions(spatial_16bit PRIVATE NR_RECEIVERS=32 NR_BITS=16 NR_RECEIVERS_PER_BLOCK=32 DEBUG=1)

target_include_directories(
    spatial_16bit
    PUBLIC
        ${PROJECT_SOURCE_DIR}/extern/tcc
        ${PROJECT_SOURCE_DIR}/extern/libpcap
        ${PROJECT_SOURCE_DIR}/extern/cudawrappers/include
        ${PROJECT_SOURCE_DIR}/include
)

set_target_properties(spatial_16bit PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

