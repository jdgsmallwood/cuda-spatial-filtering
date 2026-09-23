#pragma once

// GPU-free geometry types and coordinate-transform helpers for coherent beam
// steering: given a celestial target (J2000 RA/Dec, or "zenith"), the array's
// site, and the current time, compute the target's direction cosines (l, m, n)
// in the array's local East-North-Up frame -- the geometric input to
// compute_steering_weights()'s phase calculation (see pipeline.hpp).
//
// Lives outside common.hpp/pipeline.hpp to avoid a circular include (both
// need these types), and keeps topocentric_direction() unit-testable in
// isolation from the CUDA/pipeline machinery (see tests/test_pointing.cpp).
//
// topocentric_direction() is declared but not defined here: its
// casacore-calling implementation must be compiled by the system C++ compiler
// against the system casacore libraries -- nvcc resolves <casacore/...> to an
// ABI-incompatible conda copy, producing "undefined reference" link errors if
// casacore measures templates are instantiated in an nvcc translation unit.
// See src/pointing.cpp and src/CMakeLists.txt for the build wiring.

#include <chrono>
#include <cmath>
#include <string>

// Antenna position relative to the array reference point, in the local
// East-North-Up frame (metres). From config.json's "antenna_positions",
// keyed by absolute antenna ID.
struct ENUPosition {
  double east = 0.0;
  double north = 0.0;
  double up = 0.0;
};

// Array reference point (WGS84 lat/long/height): the ENU origin and the
// observing site for RA/Dec -> topocentric conversions.
struct ArrayLocation {
  double latitude_deg = 0.0;
  double longitude_deg = 0.0;
  double height_m = 0.0;
};

// Linear channel-to-frequency map: base_frequency_hz is the sky frequency of
// absolute channel 0, channel_bandwidth_hz is the per-channel width.
struct FrequencyPlan {
  double base_frequency_hz = 0.0;
  double channel_bandwidth_hz = 0.0;
};

// A beam's pointing target: "radec" tracks a J2000 source (re-steered
// periodically to follow sidereal motion); "zenith" is straight up and
// time-invariant.
struct BeamTarget {
  std::string mode = "zenith"; // "radec" or "zenith"
  double ra_deg = 0.0;
  double dec_deg = 0.0;
};

inline double channel_to_frequency_hz(int absolute_channel_index,
                                      const FrequencyPlan &plan) {
  return plan.base_frequency_hz + static_cast<double>(absolute_channel_index) *
                                      plan.channel_bandwidth_hz;
}

// Sky frequency of a channel in the post-PFB processing axis.  With fine
// channelisation disabled, processing_channel_index is simply a coarse-channel
// offset.  The Lambda second-stage PFB is 32/27 oversampled; after trimming two
// bins from each edge, the 28 retained bin centres span -0.5..+0.5 of one
// coarse channel in steps of 1/27.  Adjacent coarse channels therefore share
// their boundary bin.  This is the same convention used by
// lambda_commissioning.utils.fine_channel_freqs.
inline double processing_channel_to_frequency_hz(
    size_t processing_channel_index, int min_coarse_channel,
    const FrequencyPlan &plan, size_t nr_fine_channels,
    size_t fine_channel_edge_trim, double fine_oversample_denominator = 27.0) {
  if (nr_fine_channels <= 1)
    return channel_to_frequency_hz(
        min_coarse_channel + static_cast<int>(processing_channel_index), plan);

  const size_t nr_effective_fine_channels =
      nr_fine_channels - 2 * fine_channel_edge_trim;
  const size_t coarse_offset =
      processing_channel_index / nr_effective_fine_channels;
  const size_t retained_fine_index =
      processing_channel_index % nr_effective_fine_channels;
  const double offset_in_coarse_channels =
      (static_cast<double>(retained_fine_index) -
       (static_cast<double>(nr_effective_fine_channels) - 1.0) / 2.0) /
      fine_oversample_denominator;

  return channel_to_frequency_hz(
             min_coarse_channel + static_cast<int>(coarse_offset), plan) +
         offset_in_coarse_channels * plan.channel_bandwidth_hz;
}

// Direction cosines in the local East-North-Up frame:
// l = sin(az)*cos(el), m = cos(az)*cos(el), n = sin(el).
struct DirectionCosines {
  double l = 0.0;
  double m = 0.0;
  double n = 1.0;
};

// MJD (UTC) of a UTC time point, as casacore::MEpoch expects. MJD of the
// Unix epoch is 40587.0.
inline double to_mjd_utc(std::chrono::system_clock::time_point utc_time) {
  using namespace std::chrono;
  double unix_seconds =
      duration_cast<duration<double>>(utc_time.time_since_epoch()).count();
  return 40587.0 + unix_seconds / 86400.0;
}

// Zenith is straight up by definition, (l, m, n) = (0, 0, 1) -- independent
// of time and site, which is why a zenith beam never needs re-steering.
inline DirectionCosines zenith_direction() {
  return DirectionCosines{0.0, 0.0, 1.0};
}

// Converts a J2000 RA/Dec to topocentric direction cosines for an observer at
// the given WGS84 site and UTC time, via casacore's MeasFrame/MeasConvert
// (J2000 -> AzEl, folding in precession/nutation/aberration/sidereal time --
// why this must be re-evaluated periodically to keep a beam on-target).
// Defined in src/pointing.cpp; see the header comment for why.
DirectionCosines
topocentric_direction(double ra_deg, double dec_deg,
                      std::chrono::system_clock::time_point utc_time,
                      double latitude_deg, double longitude_deg,
                      double height_m);
