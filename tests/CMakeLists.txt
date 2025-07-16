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

add_executable(
  BeamformingTests_8bit test_beamforming.cpp
)
target_compile_definitions(BeamformingTests_8bit PRIVATE NR_BITS=8 NR_RECEIVERS=32)

target_link_libraries(BeamformingTests_8bit PRIVATE GTest::gtest_main spatial_8bit cudart)


add_executable(
  BeamformingTests_16bit test_beamforming.cpp
)
target_compile_definitions(BeamformingTests_16bit PRIVATE NR_BITS=16 NR_RECEIVERS=32)

target_link_libraries(BeamformingTests_16bit PRIVATE GTest::gtest_main spatial_16bit cudart)


include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(BeamformingTests_16bit)
gtest_discover_tests(BeamformingTests_8bit)
