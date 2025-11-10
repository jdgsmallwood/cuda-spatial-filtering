#include "spatial/output.hpp"
#include "spatial/pipeline.hpp"
#include "spatial/spatial.hpp"
#include <array>
#include <complex>
#include <cstring>
#include <gtest/gtest.h>
#include <vector>

struct MockPacketDataStructure {
  std::complex<float> data[1][1][1];
  float scales[1][1][1];
};

struct MockPacketScaleStructure {
  float scale;
};

struct MockPacketEntry {
  int length = 0;
  bool processed = false;
  sockaddr_in sender_addr{};
  timeval timestamp{};
  using PacketScaleStructure = MockPacketScaleStructure;
  using PacketDataStructure = MockPacketDataStructure;

  MockPacketEntry *self = this;             // For mock use
  MockPacketEntry *parse() { return this; } // Dummy parse
};

struct MockPacketFinalDataType {
  struct Samples {
    bool dummy = true;
  };
  Samples *samples = new Samples();
  bool arrivals[1][1][1][1] = {{{{false}}}};
  int buffer_index = 0;

  void zero_missing_packets() {
    // pretend to zero data
  }
};

struct MockT {
  static constexpr int NR_CHANNELS = 1;
  static constexpr int NR_FPGA_SOURCES = 1;
  static constexpr int NR_RECEIVERS_PER_PACKET = 1;
  static constexpr int NR_RECEIVERS = 1;
  static constexpr int NR_TIME_STEPS_PER_PACKET = 1;
  static constexpr int NR_POLARIZATIONS = 1;

  using PacketEntryType = MockPacketEntry;
  using PacketFinalDataType = MockPacketFinalDataType;
  using PacketDataStructure = MockPacketDataStructure;
  using PacketScaleStructure = MockPacketScaleStructure;
  using PacketSamplesType = typename MockPacketFinalDataType::Samples;
  using Sample = std::complex<float>;
};
class MockPipeline : public GPUPipeline {
public:
  bool executed = false;
  void execute_pipeline(MockT::PacketFinalDataType *d_sample) {
    executed = true;
  }
};

class ProcessorStateTest : public ::testing::Test {
protected:
  static constexpr size_t NR_PACKETS_FOR_CORR = 4;
  static constexpr size_t NR_BETWEEN_SAMPLES = 2;
  static constexpr size_t MIN_FREQ_CHANNEL = 0;

  ProcessorState<MockT> state{NR_PACKETS_FOR_CORR, NR_BETWEEN_SAMPLES,
                              MIN_FREQ_CHANNEL};
  MockPipeline pipeline;

  void SetUp() override { state.set_pipeline(&pipeline); }

  void TearDown() override {}
};

TEST_F(ProcessorStateTest, CanCallPrivateCleanup) { state.cleanup(); }

TEST_F(ProcessorStateTest, InitializationSetsDefaults) {
  EXPECT_EQ(state.buffers_initialized, false);
  EXPECT_EQ(state.current_buffer, 0);
  EXPECT_EQ(state.write_index.load(), 0);
  EXPECT_EQ(state.read_index.load(), 0);
}
