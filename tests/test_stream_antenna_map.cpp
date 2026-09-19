#include "spatial/stream_antenna_map.hpp"

#include <gtest/gtest.h>

TEST(StreamAntennaMapTest, SuppliedMapIsAuthoritative) {
  StreamAntennaMap map;

  // Input FPGA order is deliberately neither sorted nor zero-based. Antenna
  // IDs cannot have come from AntennaMapRegistry; antenna 7 swaps X and Y.
  map.entries[9][0] = {42, 0};
  map.entries[9][1] = {42, 1};
  map.entries[9][2] = {-1, 0};
  map.entries[9][3] = {-1, 1};
  map.entries[4][0] = {7, 1};
  map.entries[4][1] = {7, 0};
  map.entries[4][2] = {-1, 0};
  map.entries[4][3] = {-1, 1};

  const std::vector<int> selected_fpgas{9, 4};
  auto [recv, pol] = map.build_permutation(selected_fpgas, 2, 2);

  // Antenna 7 is on input FPGA slot 1/receiver 0, with X in raw pol 1.
  // Antenna 42 is on input FPGA slot 0/receiver 0.
  EXPECT_EQ(recv, (std::vector<int>{2, 2, 0, 0, -1, -1, -1, -1}));
  EXPECT_EQ(pol,  (std::vector<int>{1, 0, 0, 1,  0,  0,  0,  0}));

  auto canonical =
      map.build_canonical_antenna_mapping(selected_fpgas, 2, 2);
  EXPECT_EQ(canonical.at(0), 7);
  EXPECT_EQ(canonical.at(1), 42);
  EXPECT_EQ(canonical.at(2), -1);
  EXPECT_EQ(canonical.at(3), -1);
}
