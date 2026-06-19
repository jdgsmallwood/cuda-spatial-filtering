
# pointing.cpp's casacore calls must resolve <casacore/...> to the *system*
# headers (/usr/include/casacore) that match the linked libcasa_*.so -- not
# conda's older, ABI-incompatible copy under /opt/conda/include (see the
# comment block in include/spatial/pointing.hpp). It's therefore built as its
# own object library, deliberately not linked against any CUDA target or
# ${CASACORE_INCLUDE_DIRS}: both inject "-isystem /opt/conda/include"
# (directly, or transitively via HDF5_INCLUDE_DIRS in FindCasaCore.cmake) and
# g++ searches "-isystem" dirs before its built-in /usr/include, so that one
# flag would force the wrong header regardless of include-path ordering.
# pointing.cpp has no GPU or HDF5-table dependency, so simply not pulling
# those include dirs in sidesteps the clash; /usr/include is already on g++'s
# default search path.
#
# Re-running find_package(CasaCore REQUIRED) here (redundant with the
# top-level CMakeLists.txt) because add_subdirectory(src) runs first, so
# CASACORE_LIBRARIES would otherwise be unset in this scope.
find_package(CasaCore REQUIRED)

add_library(pointing OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/pointing.cpp)
target_include_directories(pointing PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(pointing PRIVATE ${CASACORE_LIBRARIES})
set_target_properties(pointing PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(spatial
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cpp
${CMAKE_CURRENT_SOURCE_DIR}/spatial.cu
${CMAKE_CURRENT_SOURCE_DIR}/ethernet.cpp
${CMAKE_CURRENT_SOURCE_DIR}/tensor.cpp
${CMAKE_CURRENT_SOURCE_DIR}/packet_formats.cpp
${CMAKE_CURRENT_SOURCE_DIR}/pipeline.cpp
$<TARGET_OBJECTS:pointing>
)


target_link_libraries(
    spatial PRIVATE cusolver cuda cudart cublas tcc CUDA::cudart ccglib pcap cutensor HighFive::HighFive spdlog::spdlog ${CFITSIO_LIBRARIES} ${CASACORE_LIBRARIES}
)


target_include_directories(
    spatial
    PUBLIC
        ${PROJECT_SOURCE_DIR}/extern/tcc
        ${PROJECT_SOURCE_DIR}/extern/libpcap
        ${PROJECT_SOURCE_DIR}/extern/cudawrappers/include
        ${PROJECT_SOURCE_DIR}/extern/spdlog/include
        ${PROJECT_SOURCE_DIR}/include
        ${CFITSIO_INCLUDE_DIR}
        ${PSRDADA_ROOT}/include
)

target_link_directories(spatial PRIVATE ${PSRDADA_ROOT}/lib)

set_target_properties(spatial PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(spatial PROPERTIES CUDA_ARCHITECTURES "native")


if(CMAKE_BUILD_TYPE MATCHES Debug)
  #target_link_libraries(spatial PRIVATE sanitizer_flags)
endif()
