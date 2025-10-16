message(STATUS "Adding Tests for CUDASpatialFiltering")
set(CTEST_OUTPUT_ON_FAILURE ON)
add_executable(
    CUDASpatialFilteringTests test_spatial.cpp
)

target_link_libraries(CUDASpatialFilteringTests
        PRIVATE 
            GTest::gtest_main
            spatial
            CUDA::cudart
            ccglib
            HighFive::HighFive
)

add_executable(
    CUDASpatialFilteringPipelineTests test_pipeline.cu
)

target_link_libraries(CUDASpatialFilteringPipelineTests
        PRIVATE 
            GTest::gtest_main
            spatial
            CUDA::cudart
            ccglib
            HighFive::HighFive
            cutensor
            tcc
)
add_executable(
  CUDASpatialFilteringCPUTests test_packet_formats.cpp
)
target_link_libraries(CUDASpatialFilteringCPUTests PRIVATE GTest::gtest_main spatial HighFive::HighFive CUDA::cudart)



include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
gtest_discover_tests(CUDASpatialFilteringPipelineTests)
