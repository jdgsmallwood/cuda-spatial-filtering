#include <cstring>
#include <filesystem>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <memory>
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

  using BeamOutputType =
      float[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS]
           [NR_PACKETS_FOR_CORRELATION * NR_TIME_STEPS_PER_PACKET][COMPLEX];
  using ArrivalOutputType =
      bool[NR_CHANNELS][NR_PACKETS_FOR_CORRELATION][NR_FPGA_SOURCES];
  using VisibilitiesOutputType =
      float[NR_CHANNELS][NR_BASELINES][NR_POLARIZATIONS][NR_POLARIZATIONS]
           [COMPLEX];
};

namespace fs = std::filesystem;

static std::string make_temp_hdf5_file() {
  auto tmp = fs::temp_directory_path() / "test_output.h5";
  if (fs::exists(tmp))
    fs::remove(tmp);
  return tmp.string();
}

TEST(HDF5BeamWriterTest, WritesBeamAndArrivalsAndSeq) {
  std::string filename = make_temp_hdf5_file();
  HighFive::File file(filename, HighFive::File::Truncate);

  HDF5BeamWriter<MockT::BeamOutputType, MockT::ArrivalOutputType> writer(file);

  // Prepare dummy data
  MockT::BeamOutputType beam_data;
  MockT::ArrivalOutputType arrival_data;

  writer.write_beam_block(&beam_data, &arrival_data,
                          /*start_seq=*/100,
                          /*end_seq=*/200);
  writer.flush();

  // Reopen and verify
  HighFive::File verify_file(filename, HighFive::File::ReadOnly);

  auto beam_ds = verify_file.getDataSet("beam_data");
  auto beam_dims = beam_ds.getDimensions();
  ASSERT_EQ(beam_dims[0], MockT::NR_CHANNELS);
  ASSERT_EQ(beam_dims[1], MockT::NR_POLARIZATIONS);

  auto arr_ds = verify_file.getDataSet("arrivals");
  auto arr_dims = arr_ds.getDimensions();
  ASSERT_EQ(arr_dims[0], 1);
  ASSERT_EQ(arr_dims[1], 1);

  auto seq_ds = verify_file.getDataSet("beam_seq_nums");
  std::vector<int> seq_out(2);
  seq_ds.select({0, 0}, {1, 2}).read(seq_out);
  EXPECT_EQ(seq_out[0], 100);
  EXPECT_EQ(seq_out[1], 200);
}
