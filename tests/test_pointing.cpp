// Tests for the pure, GPU-free coordinate-transform helpers in
// spatial/pointing.hpp -- including topocentric_direction(), whose
// casacore-calling implementation (src/pointing.cpp) must be compiled and
// linked by the system C++ compiler against the system casacore libraries
// (see the comment blocks in pointing.hpp and src/CMakeLists.txt for why
// mixing it into an nvcc translation unit produces ABI-mismatched
// "undefined reference" link errors). This file is therefore built as its
// own plain-C++ executable (tests/CMakeLists.txt: PointingTests), NOT folded
// into the nvcc-compiled PipelineTests/test_app_common infrastructure.

#include "spatial/pointing.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace {

// Tolerance for direction-cosine comparisons against astronomically-derived
// expected values. Generous enough to absorb the drift between the J2000
// mean celestial pole (the RA/Dec frame topocentric_direction's MDirection
// input is specified in) and the true celestial pole of the current date --
// precession moves the pole by roughly 20 arcsec/year (~0.15 degrees over a
// couple of decades); nutation and aberration contribute much less -- while
// still being tight enough to catch real bugs (a degrees/radians mixup or an
// azimuth/elevation swap would produce errors of order 0.1-1.0).
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
  // 2000-01-01T12:00:00 UTC is, by definition, MJD 51544.5 (the J2000.0
  // epoch); 946684800 is that calendar date's Unix timestamp at 00:00 UTC.
  using namespace std::chrono;
  auto j2000_noon =
      system_clock::time_point{} + seconds(946684800) + hours(12);
  EXPECT_NEAR(to_mjd_utc(j2000_noon), 51544.5, 1e-9);
}

TEST(PointingTest, TopocentricDirectionIsAUnitVector) {
  // l = sin(az)*cos(el), m = cos(az)*cos(el), n = sin(el), so
  // l^2 + m^2 + n^2 == cos^2(el) + sin^2(el) == 1 for *any* az/el -- i.e.
  // for any RA/Dec/time/site. A bug that mixed up units, frames, or which
  // component is which would generally break this identity.
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
  // The north celestial pole sits on Earth's rotation axis, so -- unlike
  // every other point on the sky -- it does not appear to move as the sky
  // rotates: a northern-hemisphere observer at latitude phi always sees it
  // due north (azimuth 0) at elevation phi. That gives an analytically-known,
  // time-invariant answer, (l, m, n) = (0, cos(phi), sin(phi)), to check this
  // function's output against directly -- mirroring the plan's verification
  // note: "source at the celestial pole as seen from a given latitude =>
  // direction matches latitude-derived elevation". RA is irrelevant at
  // Dec = +90 (a degenerate coordinate at the pole), so any value will do.
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
