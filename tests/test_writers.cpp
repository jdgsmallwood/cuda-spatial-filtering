#include <cstdio>
#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <memory>
#include <complex>
#include <cuda_fp16.h>
#include <vector>

#include "spatial/writers.hpp"

struct MockT {
  static constexpr size_t NR_CHANNELS = 1;
  static constexpr size_t NR_FPGA_SOURCES = 1;
  static constexpr size_t NR_BEAMS = 1;
  static constexpr size_t NR_TIME_STEPS_PER_PACKET = 8;
  static constexpr size_t NR_PACKETS_FOR_CORRELATION = 1;
  static constexpr size_t NR_POLARIZATIONS = 2;
  static constexpr size_t COMPLEX = 2;
  static constexpr size_t NR_RECEIVERS = 4;
  static constexpr size_t NR_PADDED_RECEIVERS = 32;

  static constexpr size_t NR_BASELINES =
      NR_PADDED_RECEIVERS * (NR_PADDED_RECEIVERS + 1) / 2;
  static constexpr size_t NR_BASELINES_UNPADDED =
      NR_RECEIVERS * (NR_RECEIVERS + 1) / 2;

  using BeamOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
           [NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET][COMPLEX];
  using ArrivalOutputType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES];
  using VisibilitiesOutputType =
      float[NR_CHANNELS][NR_BASELINES_UNPADDED][NR_POLARIZATIONS]
           [NR_POLARIZATIONS][COMPLEX];
};

struct AdaptiveWriterMockT {
  static constexpr size_t NR_CHANNELS = 2;
  static constexpr size_t NR_POLARIZATIONS = 2;
  static constexpr size_t NR_BEAMS = 2;
  static constexpr size_t NR_TIMES = 4;
  static constexpr size_t NR_PACKETS_FOR_CORRELATION = 3;
  static constexpr size_t NR_FPGA_SOURCES = 2;

  using BeamOutputType =
      std::complex<__half>[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS][NR_TIMES];
  using ArrivalOutputType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES];
};

namespace fs = std::filesystem;

static std::string make_temp_hdf5_file() {
  auto tmpl = fs::temp_directory_path() / "test_output_XXXXXX.h5";
  std::string s = tmpl.string();
  int fd = mkstemps(s.data(), 3); // suffix length = len(".h5") = 3
  if (fd >= 0) close(fd);         // mkstemps creates the file; HDF5 will re-open
  return s;
}

template <typename BeamT>
std::vector<uint16_t> beam_bits(const BeamT &beam_data) {
  const auto *ptr = reinterpret_cast<const uint16_t *>(&beam_data);
  return std::vector<uint16_t>(ptr, ptr + sizeof(BeamT) / sizeof(uint16_t));
}

template <typename T>
void fill_adaptive_beam_data(T &beam_data, int seed) {
  for (size_t c = 0; c < AdaptiveWriterMockT::NR_CHANNELS; ++c) {
    for (size_t p = 0; p < AdaptiveWriterMockT::NR_POLARIZATIONS; ++p) {
      for (size_t b = 0; b < AdaptiveWriterMockT::NR_BEAMS; ++b) {
        for (size_t t = 0; t < AdaptiveWriterMockT::NR_TIMES; ++t) {
          const float real = static_cast<float>(seed + 100 * c + 10 * p +
                                                3 * b + t);
          const float imag = -real - 0.5f;
          beam_data[c][p][b][t] = std::complex<__half>(__float2half(real),
                                                       __float2half(imag));
        }
      }
    }
  }
}

template <typename T>
void fill_arrivals_data(T &arrivals_data, bool seed) {
  for (size_t c = 0; c < AdaptiveWriterMockT::NR_CHANNELS; ++c) {
    for (size_t pkt = 0; pkt < AdaptiveWriterMockT::NR_PACKETS_FOR_CORRELATION;
         ++pkt) {
      for (size_t fpga = 0; fpga < AdaptiveWriterMockT::NR_FPGA_SOURCES;
           ++fpga) {
        arrivals_data[c][pkt][fpga] =
            ((c + pkt + fpga + static_cast<size_t>(seed)) % 2) == 0;
      }
    }
  }
}

TEST(HDF5BeamWriterTest, WritesBeamAndArrivalsAndSeq) {
  std::string filename = make_temp_hdf5_file();
  HighFive::File file(filename, HighFive::File::Truncate);

  HDF5BeamWriter<MockT::BeamOutputType, MockT::ArrivalOutputType> writer(file);

  // Prepare dummy data
  MockT::BeamOutputType beam_data;
  MockT::ArrivalOutputType arrival_data;

    size_t block_idx = writer.register_block(100, 200);
    void* beam_ptr = writer.get_beam_data_landing_pointer(block_idx);
    void * arrivals_ptr = writer.get_arrivals_data_landing_pointer(block_idx);

    std::memcpy(beam_ptr, &beam_data, sizeof(MockT::BeamOutputType));
    std::memcpy(arrivals_ptr, &arrival_data, sizeof(MockT::ArrivalOutputType));

    writer.register_beam_data_transfer_complete(block_idx);
    writer.register_arrivals_transfer_complete(block_idx);
    writer.drain_ready_blocks();
  writer.flush();

  // Reopen and verify
  HighFive::File verify_file(filename, HighFive::File::ReadOnly);

  auto beam_ds = verify_file.getDataSet("beam_data");
  auto beam_dims = beam_ds.getDimensions();
  ASSERT_EQ(beam_dims[0], 1); // as we only wrote one parcel of data
  ASSERT_EQ(beam_dims[1], MockT::NR_CHANNELS);
  ASSERT_EQ(beam_dims[2], MockT::NR_POLARIZATIONS);
  ASSERT_EQ(beam_dims[3], MockT::NR_BEAMS);
  ASSERT_EQ(beam_dims[4], MockT::NR_PACKETS_FOR_CORRELATION *
                              MockT::NR_TIME_STEPS_PER_PACKET);
  ASSERT_EQ(beam_dims[5], MockT::COMPLEX);

  auto arr_ds = verify_file.getDataSet("arrivals");
  auto arr_dims = arr_ds.getDimensions();
  ASSERT_EQ(arr_dims[0], 1);
  ASSERT_EQ(arr_dims[1], MockT::NR_CHANNELS);
  ASSERT_EQ(arr_dims[2], MockT::NR_PACKETS_FOR_CORRELATION);
  ASSERT_EQ(arr_dims[3], MockT::NR_FPGA_SOURCES);

  auto seq_ds = verify_file.getDataSet("beam_seq_nums");
  std::vector<std::vector<int>> seq_out(1);
  seq_ds.select({0, 0}, {1, 2}).read(seq_out);
  EXPECT_EQ(seq_out[0][0], 100);
  EXPECT_EQ(seq_out[0][1], 200);
}

TEST(HDF5BeamWriterTest, WritesAdaptiveComplexHalfBeamDataWithExpectedLayout) {
  std::string filename = make_temp_hdf5_file();
  HighFive::File file(filename, HighFive::File::Truncate);

  using BeamT = AdaptiveWriterMockT::BeamOutputType;
  using ArrivalsT = AdaptiveWriterMockT::ArrivalOutputType;
  HDF5BeamWriter<BeamT, ArrivalsT> writer(file);

  BeamT beam_data;
  ArrivalsT arrival_data;
  fill_adaptive_beam_data(beam_data, /*seed=*/7);
  fill_arrivals_data(arrival_data, /*seed=*/true);

  const size_t block_idx = writer.register_block(111, 222);
  std::memcpy(writer.get_beam_data_landing_pointer(block_idx), &beam_data,
              sizeof(BeamT));
  std::memcpy(writer.get_arrivals_data_landing_pointer(block_idx), &arrival_data,
              sizeof(ArrivalsT));
  writer.register_beam_data_transfer_complete(block_idx);
  writer.register_arrivals_transfer_complete(block_idx);
  writer.drain_ready_blocks();
  writer.flush();

  HighFive::File verify_file(filename, HighFive::File::ReadOnly);
  auto beam_ds = verify_file.getDataSet("beam_data");
  const auto dims = beam_ds.getDimensions();
  ASSERT_EQ(dims.size(), 6u);
  EXPECT_EQ(dims[0], 1u);
  EXPECT_EQ(dims[1], AdaptiveWriterMockT::NR_CHANNELS);
  EXPECT_EQ(dims[2], AdaptiveWriterMockT::NR_POLARIZATIONS);
  EXPECT_EQ(dims[3], AdaptiveWriterMockT::NR_BEAMS);
  EXPECT_EQ(dims[4], AdaptiveWriterMockT::NR_TIMES);
  EXPECT_EQ(dims[5], 2u);

  std::vector<uint16_t> stored_bits(sizeof(BeamT) / sizeof(uint16_t));
  beam_ds.read_raw(stored_bits.data());
  EXPECT_EQ(stored_bits, beam_bits(beam_data));

  auto arrivals_ds = verify_file.getDataSet("arrivals");
  const auto arrivals_dims = arrivals_ds.getDimensions();
  ASSERT_EQ(arrivals_dims.size(), 4u);
  EXPECT_EQ(arrivals_dims[0], 1u);
  EXPECT_EQ(arrivals_dims[1], AdaptiveWriterMockT::NR_CHANNELS);
  EXPECT_EQ(arrivals_dims[2], AdaptiveWriterMockT::NR_PACKETS_FOR_CORRELATION);
  EXPECT_EQ(arrivals_dims[3], AdaptiveWriterMockT::NR_FPGA_SOURCES);
}

TEST(HDF5BeamWriterTest, BatchesAdaptiveComplexHalfBeamBlocksInOrder) {
  std::string filename = make_temp_hdf5_file();
  HighFive::File file(filename, HighFive::File::Truncate);

  using BeamT = AdaptiveWriterMockT::BeamOutputType;
  using ArrivalsT = AdaptiveWriterMockT::ArrivalOutputType;
  HDF5BeamWriter<BeamT, ArrivalsT> writer(file, /*num_blocks=*/2);

  BeamT beam_a;
  BeamT beam_b;
  ArrivalsT arrivals_a;
  ArrivalsT arrivals_b;
  fill_adaptive_beam_data(beam_a, /*seed=*/3);
  fill_adaptive_beam_data(beam_b, /*seed=*/19);
  fill_arrivals_data(arrivals_a, /*seed=*/false);
  fill_arrivals_data(arrivals_b, /*seed=*/true);

  const size_t block_a = writer.register_block(10, 20);
  std::memcpy(writer.get_beam_data_landing_pointer(block_a), &beam_a,
              sizeof(BeamT));
  std::memcpy(writer.get_arrivals_data_landing_pointer(block_a), &arrivals_a,
              sizeof(ArrivalsT));
  writer.register_beam_data_transfer_complete(block_a);
  writer.register_arrivals_transfer_complete(block_a);
  writer.drain_ready_blocks();

  const size_t block_b = writer.register_block(30, 40);
  std::memcpy(writer.get_beam_data_landing_pointer(block_b), &beam_b,
              sizeof(BeamT));
  std::memcpy(writer.get_arrivals_data_landing_pointer(block_b), &arrivals_b,
              sizeof(ArrivalsT));
  writer.register_beam_data_transfer_complete(block_b);
  writer.register_arrivals_transfer_complete(block_b);
  writer.drain_ready_blocks();
  writer.flush();

  HighFive::File verify_file(filename, HighFive::File::ReadOnly);
  auto beam_ds = verify_file.getDataSet("beam_data");
  const auto dims = beam_ds.getDimensions();
  ASSERT_EQ(dims[0], 2u);

  std::vector<uint16_t> stored_bits(2 * sizeof(BeamT) / sizeof(uint16_t));
  beam_ds.read_raw(stored_bits.data());
  auto expected = beam_bits(beam_a);
  const auto beam_b_bits = beam_bits(beam_b);
  expected.insert(expected.end(), beam_b_bits.begin(), beam_b_bits.end());
  EXPECT_EQ(stored_bits, expected);

  auto seq_ds = verify_file.getDataSet("beam_seq_nums");
  std::vector<size_t> seqs(4);
  seq_ds.read_raw(seqs.data());
  ASSERT_EQ(seqs.size(), 4u);
  EXPECT_EQ(seqs[0], 10u);
  EXPECT_EQ(seqs[1], 20u);
  EXPECT_EQ(seqs[2], 30u);
  EXPECT_EQ(seqs[3], 40u);

  // Verify arrivals ordering: block a at row 0, block b at row 1
  auto arrivals_ds = verify_file.getDataSet("arrivals");
  constexpr size_t arrivals_n =
      sizeof(AdaptiveWriterMockT::ArrivalOutputType) / sizeof(bool);
  std::vector<uint8_t> stored_arrivals(2 * arrivals_n);
  arrivals_ds.read_raw(stored_arrivals.data());

  const auto *ptr_a = reinterpret_cast<const uint8_t *>(&arrivals_a);
  const auto *ptr_b = reinterpret_cast<const uint8_t *>(&arrivals_b);
  EXPECT_EQ(
      std::vector<uint8_t>(stored_arrivals.begin(),
                           stored_arrivals.begin() + arrivals_n),
      std::vector<uint8_t>(ptr_a, ptr_a + arrivals_n))
      << "arrivals for block_a not at row 0";
  EXPECT_EQ(
      std::vector<uint8_t>(stored_arrivals.begin() + arrivals_n,
                           stored_arrivals.end()),
      std::vector<uint8_t>(ptr_b, ptr_b + arrivals_n))
      << "arrivals for block_b not at row 1";
}

TEST(HDF5BeamWriterTest, WritesArrivalsValuesCorrectly) {
  std::string filename = make_temp_hdf5_file();
  HighFive::File file(filename, HighFive::File::Truncate);

  using BeamT = AdaptiveWriterMockT::BeamOutputType;
  using ArrivalsT = AdaptiveWriterMockT::ArrivalOutputType;
  HDF5BeamWriter<BeamT, ArrivalsT> writer(file);

  BeamT beam_data;
  ArrivalsT arrival_data;
  fill_adaptive_beam_data(beam_data, /*seed=*/1);
  fill_arrivals_data(arrival_data, /*seed=*/true);

  const size_t block_idx = writer.register_block(0, 1);
  std::memcpy(writer.get_beam_data_landing_pointer(block_idx), &beam_data,
              sizeof(BeamT));
  std::memcpy(writer.get_arrivals_data_landing_pointer(block_idx), &arrival_data,
              sizeof(ArrivalsT));
  writer.register_beam_data_transfer_complete(block_idx);
  writer.register_arrivals_transfer_complete(block_idx);
  writer.drain_ready_blocks();
  writer.flush();

  HighFive::File verify_file(filename, HighFive::File::ReadOnly);
  auto arrivals_ds = verify_file.getDataSet("arrivals");

  constexpr size_t n = sizeof(ArrivalsT) / sizeof(bool);
  std::vector<uint8_t> stored(n);
  arrivals_ds.read_raw(stored.data());

  const auto *expected_ptr = reinterpret_cast<const uint8_t *>(&arrival_data);
  EXPECT_EQ(stored, std::vector<uint8_t>(expected_ptr, expected_ptr + n));
}

TEST(HDF5BeamWriterTest, WritesVisibilities) {
  std::string filename = make_temp_hdf5_file();
  HighFive::File file(filename, HighFive::File::Truncate);

  const int min_channel = 98;
  const int max_channel = 105;
  HDF5VisibilitiesWriter<MockT::VisibilitiesOutputType> writer(
      file, min_channel, max_channel);

  // Prepare dummy data
  MockT::VisibilitiesOutputType vis_data;

    size_t block_idx = writer.register_block(100, 200, 0 /* missing packets */, 400 /* total packets */);
    void* vis_ptr = writer.get_visibilities_landing_pointer(block_idx);

    std::memcpy(vis_ptr, &vis_data, sizeof(MockT::VisibilitiesOutputType));
    writer.register_visibilities_transfer_complete(block_idx);
    writer.drain_ready_blocks();
  writer.flush();

  // Reopen and verify
  HighFive::File verify_file(filename, HighFive::File::ReadOnly);

  auto vis_ds = verify_file.getDataSet("visibilities");
  auto vis_dims = vis_ds.getDimensions();
  ASSERT_EQ(vis_dims[0], 1);
  ASSERT_EQ(vis_dims[1], MockT::NR_CHANNELS);
  ASSERT_EQ(vis_dims[2], MockT::NR_BASELINES_UNPADDED);
  ASSERT_EQ(vis_dims[3], MockT::NR_POLARIZATIONS);
  ASSERT_EQ(vis_dims[4], MockT::NR_POLARIZATIONS);
  ASSERT_EQ(vis_dims[5], MockT::COMPLEX);

  auto seq_ds = verify_file.getDataSet("vis_seq_nums");
  std::vector<std::vector<int>> seq_out(2);
  seq_ds.select({0, 0}, {1, 2}).read(seq_out);
  EXPECT_EQ(seq_out[0][0], 100);
  EXPECT_EQ(seq_out[0][1], 200);

  int min_channel_out, max_channel_out;
  verify_file.getAttribute("min_channel").read(min_channel_out);
  verify_file.getAttribute("max_channel").read(max_channel_out);
  EXPECT_EQ(min_channel, min_channel_out);
  EXPECT_EQ(max_channel, max_channel_out);
}
