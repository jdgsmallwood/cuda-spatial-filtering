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
  CUDASpatialFilteringCPUTests test_packet_formats.cpp
)
target_link_libraries(CUDASpatialFilteringCPUTests PRIVATE GTest::gtest_main spatial HighFive::HighFive)

include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
