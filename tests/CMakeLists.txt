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
    PipelineTests test_pipeline.cu
)

target_link_libraries(PipelineTests
        PRIVATE 
            GTest::gtest_main
            spatial
            CUDA::cudart
            ccglib
            HighFive::HighFive
            cutensor
            tcc
            CUDA::cufft
            CUDA::cusolver
)
set_target_properties(PipelineTests PROPERTIES CUDA_ARCHITECTURES "native")

add_executable(
  CUDASpatialFilteringCPUTests test_packet_formats.cpp
)
target_link_libraries(CUDASpatialFilteringCPUTests PRIVATE GTest::gtest_main spatial HighFive::HighFive CUDA::cudart)

add_executable(
  WriterTests test_writers.cpp
)
target_link_libraries(WriterTests PRIVATE GTest::gtest_main spatial HighFive::HighFive ${CASACORE_LIBRARIES} redis++ hiredis)

add_executable(
  ProcessorTests test_processor.cpp
)
target_link_libraries(ProcessorTests PRIVATE GTest::gtest_main spatial CUDA::cudart HighFive::HighFive ccglib)

add_executable(
  IncoherentSumTests test_incoherent_sum.cu
)
target_link_libraries(IncoherentSumTests PRIVATE GTest::gtest_main spatial CUDA::cudart)



add_executable(
  FoldingTests test_folding.cu
)
target_link_libraries(FoldingTests PRIVATE GTest::gtest_main spatial CUDA::cudart)

include(GoogleTest)
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
gtest_discover_tests(PipelineTests)
gtest_discover_tests(WriterTests)
gtest_discover_tests(ProcessorTests)
gtest_discover_tests(IncoherentSumTests)
gtest_discover_tests(FoldingTests)
