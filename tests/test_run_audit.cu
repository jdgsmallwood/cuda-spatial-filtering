#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <highfive/H5File.hpp>
#include <unistd.h>
#include <vector>
#include "spatial/common.hpp"

namespace fs = std::filesystem;

static std::string audit_temp(const char *suffix) {
  std::string value = (fs::temp_directory_path() /
      (std::string("run_audit_XXXXXX") + suffix)).string();
  const int fd = mkstemps(value.data(), std::strlen(suffix));
  if (fd >= 0) close(fd);
  return value;
}

TEST(HDF5RunAuditTest, EmbedsManifestAndBidirectionalMappings) {
  const std::string hdf_path = audit_temp(".h5");
  const std::string config_path = audit_temp(".json");
  { std::ofstream out(config_path); out << R"({"source":"test"})"; }
  CommonArgs args;
  args.config_filename = config_path;
  args.stream_antenna_map_filename = "missing-map.json";
  args.config = {{"normalised", 17}};
  args.fpga_id_vec = {9, 4};
  args.fpga_names = {"nic9", "nic4"};
  args.min_freq_channel = 192;
  args.canonical_recv_perm = {2, 0, -1, -1};
  args.canonical_antenna_mapping = {{0, 7}, {1, 42}, {2, -1}, {3, -1}};
  args.raw_stream_antenna_mapping = {{0, 42}, {1, 42}, {2, -1}, {3, -1},
      {4, 7}, {5, 7}, {6, -1}, {7, -1}};
  args.raw_stream_polarization_mapping = {{0, 0}, {1, 1}, {2, 0}, {3, 1},
      {4, 1}, {5, 0}, {6, 0}, {7, 1}};
  char command[] = "observe_2_2";
  char *argv[] = {command};
  { HighFive::File file(hdf_path, HighFive::File::Truncate);
    write_hdf5_run_audit(file, args, 1, argv, 2, 2); }

  HighFive::File file(hdf_path, HighFive::File::ReadOnly);
  std::string text;
  file.getDataSet("audit/run_manifest_json").read(text);
  const auto manifest = json::parse(text);
  EXPECT_EQ(manifest["command_line"], json::array({"observe_2_2"}));
  EXPECT_EQ(manifest["normalized"]["selected_fpga_ids"], json::array({9, 4}));
  EXPECT_EQ(manifest["input_files"]["config"]["content"], R"({"source":"test"})");
  EXPECT_FALSE(manifest["input_files"]["stream_antenna_map"]["present"]);
  EXPECT_TRUE(manifest["build"].contains("git_commit"));
  EXPECT_EQ(manifest["forward_mapping"].size(), 8);
  EXPECT_EQ(manifest["reverse_mapping"].size(), 8);

  auto forward = file.getDataSet("audit/forward_stream_mapping");
  EXPECT_EQ(forward.getDimensions(), (std::vector<size_t>{8, 12}));
  std::vector<int> fwd(8 * 12); forward.read_raw(fwd.data());
  EXPECT_EQ(fwd[0 * 12 + 2], 9);
  EXPECT_EQ(fwd[0 * 12 + 7], 42);
  EXPECT_EQ(fwd[0 * 12 + 8], 1);
  EXPECT_EQ(fwd[2 * 12 + 7], -1);
  EXPECT_EQ(fwd[2 * 12 + 10], 1);
  auto reverse = file.getDataSet("audit/reverse_canonical_mapping");
  EXPECT_EQ(reverse.getDimensions(), (std::vector<size_t>{8, 8}));
  std::vector<int> rev(8 * 8); reverse.read_raw(rev.data());
  EXPECT_EQ(rev[0 * 8 + 1], 7);
  EXPECT_EQ(rev[0 * 8 + 3], 5);
  EXPECT_EQ(rev[2 * 8 + 1], 42);
  EXPECT_EQ(rev[2 * 8 + 3], 0);
  EXPECT_EQ(rev[4 * 8 + 1], -1);
  EXPECT_EQ(rev[4 * 8 + 3], -1);
  EXPECT_EQ(rev[4 * 8 + 7], 1);
  fs::remove(config_path);
}
