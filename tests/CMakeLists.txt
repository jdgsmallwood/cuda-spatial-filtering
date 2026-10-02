message(STATUS "Adding Tests for CUDASpatialFiltering")
set(CTEST_OUTPUT_ON_FAILURE ON)
find_package(Python3 COMPONENTS Interpreter REQUIRED)

set(TEST_CUDA_ENV "")
if(CUDAToolkit_INCLUDE_DIRS)
  list(GET CUDAToolkit_INCLUDE_DIRS 0 FIRST_CUDA_INCLUDE_DIR)
  cmake_path(NORMAL_PATH FIRST_CUDA_INCLUDE_DIR OUTPUT_VARIABLE FIRST_CUDA_INCLUDE_DIR)
  if(FIRST_CUDA_INCLUDE_DIR MATCHES "/targets/[^/]+/include$")
    cmake_path(GET FIRST_CUDA_INCLUDE_DIR PARENT_PATH TEST_CUDA_ROOT)
  elseif(FIRST_CUDA_INCLUDE_DIR MATCHES "/include$")
    cmake_path(GET FIRST_CUDA_INCLUDE_DIR PARENT_PATH TEST_CUDA_ROOT)
  endif()
  if(TEST_CUDA_ROOT)
    set(TEST_CUDA_ENV
        "CUDA_HOME=${TEST_CUDA_ROOT};CUDA_PATH=${TEST_CUDA_ROOT}")
  endif()
endif()

add_library(test_app_common INTERFACE)
target_include_directories(test_app_common INTERFACE ${PSRDADA_ROOT}/include)
target_link_directories(test_app_common INTERFACE ${PSRDADA_ROOT}/lib)

target_compile_options(test_app_common INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=128,2417>)
target_link_libraries(test_app_common INTERFACE GTest::gtest_main spatial CUDA::cudart ccglib
HighFive::HighFive CUDA::cufft CUDA::cusolver cutensor tcc filter psrdada ${CASACORE_LIBRARIES} redis++ hiredis argparse)

set_target_properties(test_app_common PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
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
  CUDASpatialFilteringCPUTests test_packet_formats.cpp test_stream_antenna_map.cpp
)
target_link_libraries(CUDASpatialFilteringCPUTests PRIVATE test_app_common)

add_executable(
  WriterTests test_writers.cpp
)
target_link_libraries(WriterTests PRIVATE test_app_common)

add_executable(
  RunAuditTests test_run_audit.cu
)
target_link_libraries(RunAuditTests PRIVATE test_app_common)
set_target_properties(RunAuditTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
add_executable(
  ProcessorTests test_processor.cu
)
target_link_libraries(ProcessorTests PRIVATE test_app_common)

add_executable(ProcessorDirectNicTests test_direct_nic_workers.cu)
target_link_libraries(ProcessorDirectNicTests PRIVATE test_support)
target_compile_definitions(ProcessorDirectNicTests PRIVATE
    SPATIAL_PER_NIC_RING=1 SPATIAL_DIRECT_NIC_WORKERS=1)

add_executable(ProcessorDirectNicBatchTests test_direct_nic_workers.cu)
target_link_libraries(ProcessorDirectNicBatchTests PRIVATE test_support)
target_compile_definitions(ProcessorDirectNicBatchTests PRIVATE
    SPATIAL_PER_NIC_RING=1 SPATIAL_DIRECT_NIC_WORKERS=1
    SPATIAL_BATCH_PACKET_WATERMARKS=1)

add_executable(ProcessorDirectNicBatchOptimizedTests test_direct_nic_workers.cu)
target_link_libraries(ProcessorDirectNicBatchOptimizedTests PRIVATE test_support)
target_compile_definitions(ProcessorDirectNicBatchOptimizedTests PRIVATE
    SPATIAL_PER_NIC_RING=1 SPATIAL_DIRECT_NIC_WORKERS=1
    SPATIAL_BATCH_PACKET_WATERMARKS=1 SPATIAL_BATCH_NT_FENCE=1
    SPATIAL_TOUCHED_WATERMARKS=1)

add_executable(ProcessorDirectNicBatchOptimizedBatchedTests test_direct_nic_workers.cu)
target_link_libraries(ProcessorDirectNicBatchOptimizedBatchedTests PRIVATE test_support)
target_compile_definitions(ProcessorDirectNicBatchOptimizedBatchedTests PRIVATE
    SPATIAL_PER_NIC_RING=1 SPATIAL_DIRECT_NIC_WORKERS=1
    SPATIAL_BATCH_PACKET_WATERMARKS=1 SPATIAL_BATCH_NT_FENCE=1
    SPATIAL_TOUCHED_WATERMARKS=1 SPATIAL_DIRECT_BATCH_TARGET=192
    SPATIAL_DIRECT_BATCH_WAIT_US=200)

add_executable(ProcessorDirectNicTraceTests test_direct_nic_workers.cu)
target_link_libraries(ProcessorDirectNicTraceTests PRIVATE test_support)
target_compile_definitions(ProcessorDirectNicTraceTests PRIVATE
    SPATIAL_PER_NIC_RING=1 SPATIAL_DIRECT_NIC_WORKERS=1
    SPATIAL_BATCH_PACKET_WATERMARKS=1 SPATIAL_BATCH_NT_FENCE=1
    SPATIAL_TOUCHED_WATERMARKS=1 SPATIAL_DIRECT_BATCH_TARGET=192
    SPATIAL_DIRECT_BATCH_WAIT_US=200 SPATIAL_TIMING_DIAGNOSTICS=1
    SPATIAL_INGEST_TRACE=1 SPATIAL_COORDINATOR_AFFINITY=1)

add_executable(IngestTraceTests test_ingest_trace.cpp)
target_include_directories(IngestTraceTests PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(IngestTraceTests PRIVATE GTest::gtest_main)

add_executable(ProcessorDecoupledTests test_direct_nic_workers.cu test_decoupled_ingest.cu)
target_link_libraries(ProcessorDecoupledTests PRIVATE test_support)
target_compile_definitions(ProcessorDecoupledTests PRIVATE
    SPATIAL_PER_NIC_RING=1 SPATIAL_DIRECT_NIC_WORKERS=1
    SPATIAL_DECOUPLED_COMPLETION=1 SPATIAL_TIMING_DIAGNOSTICS=1
    SPATIAL_INGEST_TRACE=1)

add_executable(
  IbverbsPlacementTests test_ibverbs_placement.cu
)
target_link_libraries(IbverbsPlacementTests PRIVATE test_app_common)


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
set_target_properties(PulsarFoldTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(
  CorrBeamPipelineTests test_corr_beam_pipeline.cu
)
target_link_libraries(CorrBeamPipelineTests PRIVATE test_support)
set_target_properties(CorrBeamPipelineTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(
  StarspinTests test_starspin_pipeline.cu
)
target_link_libraries(StarspinTests PRIVATE test_support)
set_target_properties(StarspinTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(
  AdaptivePipelineTests test_adaptive_pipeline.cu
)
target_link_libraries(AdaptivePipelineTests PRIVATE test_support)
set_target_properties(AdaptivePipelineTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineDelayTests test_fine_delay.cu)
target_link_libraries(FineDelayTests PRIVATE test_support)
set_target_properties(FineDelayTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(StreamReorderTests test_stream_reorder.cu)
target_link_libraries(StreamReorderTests PRIVATE test_support)
set_target_properties(StreamReorderTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(AntennaOrderingTests test_antenna_ordering.cu)
target_link_libraries(AntennaOrderingTests PRIVATE test_support)
set_target_properties(AntennaOrderingTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(GpuDirectScatterTests test_gpudirect_scatter.cu)
target_link_libraries(GpuDirectScatterTests PRIVATE test_support)
set_target_properties(GpuDirectScatterTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizerTests test_fine_channelizer.cu)
target_link_libraries(FineChannelizerTests PRIVATE test_support)
set_target_properties(FineChannelizerTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedAntennaSpectraTests test_fine_channelized_antenna_spectra.cu)
target_link_libraries(FineChannelizedAntennaSpectraTests PRIVATE test_support)
set_target_properties(FineChannelizedAntennaSpectraTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedBeamformedSpectraTests test_fine_channelized_beamformed_spectra.cu)
target_link_libraries(FineChannelizedBeamformedSpectraTests PRIVATE test_support)
set_target_properties(FineChannelizedBeamformedSpectraTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedProjectionTests test_fine_channelized_projection.cu)
target_link_libraries(FineChannelizedProjectionTests PRIVATE test_support)
set_target_properties(FineChannelizedProjectionTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedAdaptiveTests test_fine_channelized_adaptive.cu)
target_link_libraries(FineChannelizedAdaptiveTests PRIVATE test_support)
set_target_properties(FineChannelizedAdaptiveTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedPulsarFoldTests test_fine_channelized_pulsar_fold.cu)
target_link_libraries(FineChannelizedPulsarFoldTests PRIVATE test_support)
set_target_properties(FineChannelizedPulsarFoldTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(CoarsePfbDerippleTests test_coarse_pfb_deripple.cu)
target_link_libraries(CoarsePfbDerippleTests PRIVATE test_app_common)
set_target_properties(CoarsePfbDerippleTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedCorrBeamTests test_fine_channelized_corr_beam.cu)
target_link_libraries(FineChannelizedCorrBeamTests PRIVATE test_support)
set_target_properties(FineChannelizedCorrBeamTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

add_executable(FineChannelizedStarspinTests test_fine_channelized_starspin.cu)
target_link_libraries(FineChannelizedStarspinTests PRIVATE test_support)
set_target_properties(FineChannelizedStarspinTests PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

# Standalone processor-path benchmark (not a gtest — run directly).
# Measures process_all_available_packets() throughput plus hash-lookup,
# gettimeofday, and pool-alloc microbenchmarks for the six hot-path fixes.
add_executable(benchmark_processor benchmark_processor.cu)
target_link_libraries(benchmark_processor PRIVATE test_app_common)
set_target_properties(benchmark_processor PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

include(GoogleTest)
gtest_discover_tests(IngestTraceTests)
gtest_discover_tests(ProcessorDecoupledTests TEST_PREFIX "Decoupled.")
gtest_discover_tests(CUDASpatialFilteringTests)
gtest_discover_tests(CUDASpatialFilteringCPUTests)
gtest_discover_tests(PipelineTests DISCOVERY_TIMEOUT 60)
gtest_discover_tests(WriterTests)
gtest_discover_tests(ProcessorTests)
gtest_discover_tests(ProcessorDirectNicTests)
gtest_discover_tests(ProcessorDirectNicBatchTests TEST_PREFIX "Batch.")
gtest_discover_tests(ProcessorDirectNicBatchOptimizedTests TEST_PREFIX "Optimized.")
gtest_discover_tests(ProcessorDirectNicBatchOptimizedBatchedTests TEST_PREFIX "OptimizedBatched.")
gtest_discover_tests(ProcessorDirectNicTraceTests TEST_PREFIX "Trace.")
gtest_discover_tests(IbverbsPlacementTests)
gtest_discover_tests(PointingTests)
gtest_discover_tests(PipelineHarnessSelfTest)
gtest_discover_tests(PulsarFoldTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(CorrBeamPipelineTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(StarspinTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")

# Run the same visibility/value tests with registered huge-page input buffers.
add_executable(StarspinTHPTests test_starspin_pipeline.cu)
target_link_libraries(StarspinTHPTests PRIVATE test_support)
target_compile_definitions(StarspinTHPTests PRIVATE
    SPATIAL_THP_INPUT_BUFFERS=1
    SPATIAL_THP_OUTPUT_BUFFERS=1)
gtest_discover_tests(StarspinTHPTests TEST_PREFIX "THP."
  PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(RunAuditTests)
gtest_discover_tests(AdaptivePipelineTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineDelayTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(StreamReorderTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(AntennaOrderingTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(GpuDirectScatterTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizerTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizedAntennaSpectraTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizedBeamformedSpectraTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizedProjectionTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizedAdaptiveTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizedPulsarFoldTests
  DISCOVERY_TIMEOUT 60
  PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(CoarsePfbDerippleTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
add_test(NAME CoarsePfbDeripplePythonTests
  COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/test_coarse_pfb_deripple.py)
gtest_discover_tests(FineChannelizedCorrBeamTests PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
gtest_discover_tests(FineChannelizedStarspinTests
  DISCOVERY_TIMEOUT 60
  PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")

add_test(
  NAME AdaptiveDetectionReferencePythonTests
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/scripts/adaptive_detection_reference.py
          --self-test)

add_test(
  NAME AdaptiveDetectionParityPythonCheck
  COMMAND ${Python3_EXECUTABLE}
          ${PROJECT_SOURCE_DIR}/scripts/check_adaptive_detection_parity.py
          --adaptive-test-bin $<TARGET_FILE:AdaptivePipelineTests>)
if(TEST_CUDA_ENV)
  set_tests_properties(
    AdaptiveDetectionParityPythonCheck
    PROPERTIES ENVIRONMENT "${TEST_CUDA_ENV}")
endif()
