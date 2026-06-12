// Tests for the pure, GPU-free coordinate-transform helpers in
// spatial/pointing.hpp. Built as its own plain-C++ executable (PointingTests
// in tests/CMakeLists.txt) rather than folded into the nvcc-compiled
// PipelineTests, since topocentric_direction()'s casacore-calling
// implementation must be compiled by the system C++ compiler -- see the
// comment blocks in pointing.hpp and src/CMakeLists.txt.

#include "spatial/pointing.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

// Loose enough to absorb J2000-to-current-date precession/nutation drift
// (~0.15 deg over a couple of decades), tight enough to catch a real bug like
// a degrees/radians mixup or an azimuth/elevation swap (errors of 0.1-1.0).
constexpr double kDirectionCosineTolerance = 0.01;

} // namespace

TEST(PointingTest, ZenithDirectionIsStraightUp) {
  DirectionCosines dc = zenith_direction();
  EXPECT_DOUBLE_EQ(dc.l, 0.0);
  EXPECT_DOUBLE_EQ(dc.m, 0.0);
  EXPECT_DOUBLE_EQ(dc.n, 1.0);
}

TEST(PointingTest, ChannelToFrequencyHzIsLinearInChannelIndex) {
  FrequencyPlan plan{/*base_frequency_hz=*/1.0e9,
                     /*channel_bandwidth_hz=*/1.0e5};
  EXPECT_DOUBLE_EQ(channel_to_frequency_hz(0, plan), 1.0e9);
  EXPECT_DOUBLE_EQ(channel_to_frequency_hz(10, plan), 1.0e9 + 10 * 1.0e5);
  EXPECT_DOUBLE_EQ(channel_to_frequency_hz(-3, plan), 1.0e9 - 3 * 1.0e5);
}

TEST(PointingTest, ToMjdUtcUnixEpochIs40587) {
  EXPECT_DOUBLE_EQ(to_mjd_utc(std::chrono::system_clock::time_point{}),
                   40587.0);
}

TEST(PointingTest, ToMjdUtcMatchesJ2000Epoch) {
  // 2000-01-01T12:00:00 UTC is MJD 51544.5 by definition (the J2000.0 epoch);
  // 946684800 is that date's Unix timestamp at 00:00 UTC.
  using namespace std::chrono;
  auto j2000_noon = system_clock::time_point{} + seconds(946684800) + hours(12);
  EXPECT_NEAR(to_mjd_utc(j2000_noon), 51544.5, 1e-9);
}

TEST(PointingTest, TopocentricDirectionIsAUnitVector) {
  // l^2+m^2+n^2 == cos^2(el)+sin^2(el) == 1 for any az/el, i.e. any
  // RA/Dec/time/site -- a units/frame/component mixup would break this.
  using namespace std::chrono;
  const auto utc_time = system_clock::now();

  for (double ra_deg : {0.0, 83.633, 180.0, 279.23}) {
    for (double dec_deg : {-60.0, -16.7, 0.0, 38.78, 89.0}) {
      DirectionCosines dc = topocentric_direction(
          ra_deg, dec_deg, utc_time, /*latitude_deg=*/52.91,
          /*longitude_deg=*/6.87, /*height_m=*/30.0);
      double norm_sq = dc.l * dc.l + dc.m * dc.m + dc.n * dc.n;
      EXPECT_NEAR(norm_sq, 1.0, 1e-9)
          << "ra_deg=" << ra_deg << " dec_deg=" << dec_deg;
    }
  }
}

TEST(PointingTest, NorthCelestialPoleElevationMatchesObserverLatitude) {
  // The north celestial pole sits on Earth's rotation axis, so a
  // northern-hemisphere observer at latitude phi always sees it due north at
  // elevation phi -- an analytically-known, time-invariant answer
  // (l, m, n) = (0, cos(phi), sin(phi)) to check against directly. RA is
  // irrelevant at Dec = +90 (degenerate at the pole), so any value will do.
  using namespace std::chrono;
  const double latitude_deg = 40.0;
  const double latitude_rad = latitude_deg * M_PI / 180.0;
  const DirectionCosines expected{/*l=*/0.0, /*m=*/std::cos(latitude_rad),
                                  /*n=*/std::sin(latitude_rad)};

  const auto now = system_clock::now();
  for (auto offset : {hours(0), hours(6), hours(12), hours(18)}) {
    DirectionCosines dc = topocentric_direction(
        /*ra_deg=*/123.45, /*dec_deg=*/90.0, now + offset, latitude_deg,
        /*longitude_deg=*/-71.06, /*height_m=*/0.0);

    EXPECT_NEAR(dc.l, expected.l, kDirectionCosineTolerance)
        << "offset_hours=" << offset.count();
    EXPECT_NEAR(dc.m, expected.m, kDirectionCosineTolerance)
        << "offset_hours=" << offset.count();
    EXPECT_NEAR(dc.n, expected.n, kDirectionCosineTolerance)
        << "offset_hours=" << offset.count();
  }
}
