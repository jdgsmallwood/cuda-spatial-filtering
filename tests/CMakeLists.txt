message(STATUS "Adding Tests for CUDASpatialFiltering")
set(CTEST_OUTPUT_ON_FAILURE ON)
add_executable(
    CUDASpatialFilteringTests test_spatial.cpp
)

target_link_libraries(CUDASpatialFilteringTests
        PRIVATE 
            GTest::gtest_main
            spatial
            cudart
            ccglib
)

include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)