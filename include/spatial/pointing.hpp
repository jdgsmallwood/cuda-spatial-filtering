#pragma once

// Pure, GPU-free geometry types and coordinate-transform helpers for
// coherent beam steering.
//
// Given a celestial target (J2000 RA/Dec, or "zenith") and the array's
// observing site and the current time, these functions compute the target's
// direction cosines (l, m, n) in the array's local East-North-Up (ENU)
// tangent frame -- the geometric input to the steering-phase calculation
// `phase = -2*pi*f/c * (l*east + m*north + n*up)` used by
// compute_steering_weights() (see pipeline.hpp).
//
// The small description structs below (ENUPosition, ArrayLocation,
// FrequencyPlan, BeamTarget) live here -- rather than in common.hpp, which
// includes pipeline.hpp near its top, before its own struct definitions --
// so that both common.hpp's CommonArgs (which stores parsed config.json
// geometry) and pipeline.hpp's compute_steering_weights() (which consumes
// it) can see them without a circular include.
//
// This header has no dependency on the project's CUDA/pipeline machinery, so
// the transform can be exercised and unit-tested in complete isolation (see
// tests/test_pointing.cpp).
//
// IMPORTANT -- why topocentric_direction() is only DECLARED here, not defined:
// its implementation (in src/pointing.cpp) calls into casacore's measures
// library, whose template classes (e.g. casacore::Vector, casacore::Array)
// have an ABI that depends on which compiler/standard-library headers resolve
// the casacore includes. This project's nvcc auto-detects conda's bundled g++
// as its host compiler, which resolves <casacore/...> to an older, ABI-
// incompatible copy under /opt/conda/include -- while the project links
// against the system's casacore libraries (built against /usr/include,
// CASACORE_INCLUDE_DIR). Instantiating casacore measures templates in any
// translation unit nvcc compiles therefore produces "undefined reference"
// link errors. Keeping this header free of casacore includes, and compiling
// its one casacore-calling function in src/pointing.cpp via the system
// CMAKE_CXX_COMPILER (consistent headers + libs), sidesteps the mismatch
// entirely -- see src/CMakeLists.txt for the corresponding build wiring.

#include <chrono>
#include <cmath>
#include <string>

// Antenna position relative to the array reference point, in the local
// East-North-Up tangent frame (metres). Sourced from config.json's
// "antenna_positions" block, keyed by absolute antenna ID.
struct ENUPosition {
  double east = 0.0;
  double north = 0.0;
  double up = 0.0;
};

// Lat/Long/height of the array's reference point (degrees, degrees, metres),
// used as the origin of the ENU frame and as the observing site for
// RA/Dec -> topocentric direction conversions.
struct ArrayLocation {
  double latitude_deg = 0.0;
  double longitude_deg = 0.0;
  double height_m = 0.0;
};

// Maps absolute channel indices to sky (RF) frequency in Hz -- needed to
// compute geometric steering phases (phase = -2*pi*f/c * path_length).
// Sourced from config.json's "frequency_plan" block: base_frequency_hz is
// the sky frequency of absolute channel 0 and channel_bandwidth_hz is the
// (constant) width of each channel. Defaults to all-zero placeholders until
// the instrument's real frequency plan is supplied there.
struct FrequencyPlan {
  double base_frequency_hz = 0.0;
  double channel_bandwidth_hz = 0.0;
};

// A single beam's pointing target. "radec" tracks a celestial source by its
// J2000 RA/Dec (re-steered periodically to follow sidereal motion); "zenith"
// points straight up in the local ENU frame -- a fixed direction that never
// needs re-steering.
struct BeamTarget {
  std::string mode = "zenith"; // "radec" or "zenith"
  double ra_deg = 0.0;
  double dec_deg = 0.0;
};

// Sky (RF) frequency in Hz of an absolute channel index, given the array's
// frequency plan: base_frequency_hz is the frequency of absolute channel 0
// and channel_bandwidth_hz is the (constant) width of each channel. This is
// the frequency `f` used in the geometric steering phase
// `phase = -2*pi*f/c * (direction . antenna_position)`.
inline double channel_to_frequency_hz(int absolute_channel_index,
                                      const FrequencyPlan &plan) {
  return plan.base_frequency_hz +
         static_cast<double>(absolute_channel_index) *
             plan.channel_bandwidth_hz;
}

// Direction cosines of a target in the array's local East-North-Up frame:
// l = sin(az)*cos(el), m = cos(az)*cos(el), n = sin(el). A source directly
// overhead (zenith) has (l, m, n) = (0, 0, 1); a source on the horizon due
// east has (l, m, n) = (1, 0, 0).
struct DirectionCosines {
  double l = 0.0;
  double m = 0.0;
  double n = 1.0;
};

// Modified Julian Date (UTC, fractional days) of a UTC time point -- the time
// representation casacore::MEpoch(Quantity, MEpoch::UTC) expects. The MJD of
// the Unix epoch (1970-01-01T00:00:00 UTC) is 40587.0.
inline double to_mjd_utc(std::chrono::system_clock::time_point utc_time) {
  using namespace std::chrono;
  double unix_seconds =
      duration_cast<duration<double>>(utc_time.time_since_epoch()).count();
  return 40587.0 + unix_seconds / 86400.0;
}

// Direction cosines of the local zenith: by definition straight up in the
// array's ENU frame, (l, m, n) = (0, 0, 1) -- independent of time, latitude,
// longitude, and height. This is also why a zenith-pointed beam never needs
// re-steering: its direction cosines are time-invariant, so they can be
// computed once (here, or via target_direction()/is_time_invariant() below)
// and reused for the lifetime of the run.
inline DirectionCosines zenith_direction() { return DirectionCosines{0.0, 0.0, 1.0}; }

// Converts a J2000 RA/Dec to topocentric direction cosines (l, m, n) in the
// array's local East-North-Up frame, for an observer at the given site
// (WGS84 latitude/longitude/height) at the given UTC time.
//
// Uses casacore::MeasFrame + MeasConvert to perform the J2000 -> AzEl
// transform (this correctly folds in precession, nutation, aberration, polar
// motion, and sidereal time -- exactly the effects that make a fixed-RA/Dec
// source drift across a topocentric frame over time, and the reason this
// function must be re-evaluated periodically to keep a beam on-target), then
// converts the resulting (azimuth, elevation) to ENU direction cosines via
// l = sin(az)*cos(el), m = cos(az)*cos(el), n = sin(el).
//
// Defined in src/pointing.cpp (NOT inline here) -- see the comment block
// above the #include list for why this function's casacore-calling body must
// live in a translation unit compiled by the system C++ compiler.
DirectionCosines topocentric_direction(double ra_deg, double dec_deg,
                                       std::chrono::system_clock::time_point utc_time,
                                       double latitude_deg, double longitude_deg,
                                       double height_m);
