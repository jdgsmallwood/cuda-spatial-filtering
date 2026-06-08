message(STATUS "Adding Tests for CUDASpatialFiltering")
set(CTEST_OUTPUT_ON_FAILURE ON)

add_library(test_app_common INTERFACE)
target_include_directories(test_app_common INTERFACE ${PSRDADA_ROOT}/include)
target_link_directories(test_app_common INTERFACE ${PSRDADA_ROOT}/lib)

target_compile_options(test_app_common INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128,2417>)
target_link_libraries(test_app_common INTERFACE GTest::gtest_main spatial CUDA::cudart ccglib
HighFive::HighFive CUDA::cufft CUDA::cusolver cutensor tcc psrdada ${CASACORE_LIBRARIES} redis++ hiredis)

set_target_properties(test_app_common PROPERTIES CUDA_ARCHITECTURES "native")
add_executable(
    CUDASpatialFilteringTests test_spatial.cpp
)

target_link_libraries(CUDASpatialFilteringTests
        PRIVATE  test_app_common
)

add_executable(
    PipelineTests test_pipeline.cu
)

target_link_libraries(PipelineTests
        PRIVATE test_app_common
)

add_executable(
  CUDASpatialFilteringCPUTests test_packet_formats.cpp
)
target_link_libraries(CUDASpatialFilteringCPUTests PRIVATE test_app_common)

add_executable(
  WriterTests test_writers.cpp
)
target_link_libraries(WriterTests PRIVATE test_app_common)

add_executable(
  ProcessorTests test_processor.cu
)
target_link_libraries(ProcessorTests PRIVATE test_app_common)

# PointingTests covers spatial/pointing.hpp's topocentric_direction(), whose
# casacore-calling implementation (src/pointing.cpp, linked in here via the
# `pointing` OBJECT library defined in src/CMakeLists.txt) must be compiled
# and linked by the system C++ compiler against the system casacore libraries
# -- mirroring that target's wiring exactly, and deliberately NOT linking
# test_app_common (whose CUDA::cudart/ccglib/cutensor/... dependencies inject
# "-isystem /opt/conda/include", resolving <casacore/...> to an
# ABI-incompatible header copy and producing "undefined reference" link
# errors -- see the comment blocks in pointing.hpp and src/CMakeLists.txt).
add_executable(
  PointingTests test_pointing.cpp $<TARGET_OBJECTS:pointing>
)
target_include_directories(PointingTests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(PointingTests PRIVATE GTest::gtest_main ${CASACORE_LIBRARIES})

include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
gtest_discover_tests(PipelineTests)
gtest_discover_tests(WriterTests)
gtest_discover_tests(ProcessorTests)
gtest_discover_tests(PointingTests)
