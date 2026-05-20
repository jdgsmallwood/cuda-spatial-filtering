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

include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
gtest_discover_tests(PipelineTests)
gtest_discover_tests(WriterTests)
gtest_discover_tests(ProcessorTests)
