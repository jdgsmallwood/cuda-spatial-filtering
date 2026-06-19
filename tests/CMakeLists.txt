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


# Built standalone (not linked against test_app_common) so pointing.cpp's
# casacore calls are compiled/linked entirely by the system C++ compiler --
# see the comment blocks in pointing.hpp and src/CMakeLists.txt for why mixing
# casacore with the CUDA/conda toolchain produces ABI-mismatched link errors.
add_executable(
  PointingTests test_pointing.cpp $<TARGET_OBJECTS:pointing>
)
target_include_directories(PointingTests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(PointingTests PRIVATE GTest::gtest_main ${CASACORE_LIBRARIES})

# Shared high-level pipeline-test harness (tests/support/*.hpp): synthetic
# packet builders, canonical small LambdaConfigs, a SyntheticPipelineRun driver
# that wires a real ProcessorState to a real GPUPipeline via the synchronous
# pipeline seam, and property-based assertion helpers. Header-only -- this
# INTERFACE library just adds the include path so `#include "support/..."`
# resolves, and pulls in test_app_common's dependencies.
add_library(test_support INTERFACE)
target_include_directories(test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_support INTERFACE test_app_common)

add_executable(
  PipelineHarnessSelfTest support/test_harness_selftest.cu
)
target_link_libraries(PipelineHarnessSelfTest PRIVATE test_support)

add_executable(
  PulsarFoldTests test_pulsar_fold.cu
)
target_link_libraries(PulsarFoldTests PRIVATE test_support)
set_target_properties(PulsarFoldTests PROPERTIES CUDA_ARCHITECTURES "native")

# Standalone processor-path benchmark (not a gtest — run directly).
# Measures process_all_available_packets() throughput plus hash-lookup,
# gettimeofday, and pool-alloc microbenchmarks for the six hot-path fixes.
add_executable(benchmark_processor benchmark_processor.cu)
target_link_libraries(benchmark_processor PRIVATE test_app_common)
set_target_properties(benchmark_processor PROPERTIES CUDA_ARCHITECTURES "native")

include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
gtest_discover_tests(PipelineTests)
gtest_discover_tests(WriterTests)
gtest_discover_tests(ProcessorTests)
gtest_discover_tests(PointingTests)
gtest_discover_tests(PipelineHarnessSelfTest)
gtest_discover_tests(PulsarFoldTests)
