
# pointing.cpp calls into casacore's measures library (topocentric_direction(),
# declared in spatial/pointing.hpp). It must be compiled against the *system*
# casacore headers (/usr/include/casacore, matching CASACORE_INCLUDE_DIR) --
# the ones the project links (libcasa_*.so under /usr/lib/x86_64-linux-gnu) --
# NOT the older, ABI-incompatible copy conda bundles under /opt/conda/include
# (see the comment block in include/spatial/pointing.hpp for why mixing them
# produces "undefined reference to casacore::Vector<double>::Vector()" et al.
# at link time).
#
# It is built as its OWN object library, deliberately NOT linked against any
# CUDA target (cudart/CUDA::cudart/cusolver/...), because those targets'
# INTERFACE_INCLUDE_DIRECTORIES inject "-isystem /opt/conda/include" -- and
# since g++ searches all "-isystem" directories as one ordered group *before*
# its built-in system directories (which is where /usr/include implicitly
# lives, as CASACORE_INCLUDE_DIR), that single flag would make <casacore/...>
# resolve under /opt/conda/include regardless of how CASACORE_INCLUDE_DIRS is
# ordered in this target's own include path -- there is no include-order flag
# that out-prioritises it once it's present. pointing.cpp has no GPU
# dependency at all (pointing.hpp is explicitly "GPU-free"), so simply never
# pulling CUDA's include dirs into its compile command sidesteps the clash
# entirely.
#
# find_package(CasaCore REQUIRED) is re-run here (redundantly with the
# top-level CMakeLists.txt) because add_subdirectory(src) runs before the
# top-level find_package(CasaCore), so CASACORE_LIBRARIES/CASACORE_INCLUDE_DIRS
# would otherwise be unset in this scope.
find_package(CasaCore REQUIRED)

# Deliberately NOT adding ${CASACORE_INCLUDE_DIRS} here: FindCasaCore.cmake
# transitively appends HDF5_INCLUDE_DIRS (to support casacore's "casa" table
# component) -- which in this conda-based environment resolves to
# /opt/conda/include, reintroducing exactly the ABI-mismatched <casacore/...>
# header resolution this object library exists to avoid (see comment above).
# pointing.cpp only touches casacore's Measures API (no HDF5-backed types), and
# /usr/include -- where the system casacore headers live, CASACORE_INCLUDE_DIR
# -- is already on g++'s default system include path, exactly as verified by
# the standalone compile/link/run check during development of pointing.hpp.
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
