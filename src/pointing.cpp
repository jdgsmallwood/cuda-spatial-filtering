#include "spatial/pointing.hpp"

// The only translation unit that includes casacore's measures headers --
// compiled by the system C++ compiler so its casacore ABI matches the linked
// libraries. See the comment block in pointing.hpp.
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/casa/Quanta/MVPosition.h>
#include <casacore/casa/Quanta/Quantum.h>
#include <casacore/measures/Measures/MCDirection.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/measures/Measures/MEpoch.h>
#include <casacore/measures/Measures/MPosition.h>
#include <casacore/measures/Measures/MeasConvert.h>
#include <casacore/measures/Measures/MeasFrame.h>

DirectionCosines
topocentric_direction(double ra_deg, double dec_deg,
                      std::chrono::system_clock::time_point utc_time,
                      double latitude_deg, double longitude_deg,
                      double height_m) {
  using namespace casacore;

  MPosition site(MVPosition(Quantity(height_m, "m"),
                            Quantity(longitude_deg, "deg"),
                            Quantity(latitude_deg, "deg")),
                 MPosition::WGS84);
  MEpoch epoch(Quantity(to_mjd_utc(utc_time), "d"), MEpoch::UTC);
  MeasFrame frame(epoch, site);

  MDirection target(Quantity(ra_deg, "deg"), Quantity(dec_deg, "deg"),
                    MDirection::J2000);
  // Antenna ENU uses the WGS84 surface normal, not geocentric up.
  MDirection::Ref azel_ref(MDirection::AZELGEO, frame);
  MDirection azel = MDirection::Convert(target, azel_ref)();

  Vector<Double> azel_rad = azel.getAngle("rad").getValue();
  double az = azel_rad[0];
  double el = azel_rad[1];
  std::cout << "Updating Beam weights. " << "Az is " << az << " and el is "
            << el << std::endl;

  DirectionCosines dc;
  dc.l = std::sin(az) * std::cos(el);
  dc.m = std::cos(az) * std::cos(el);
  dc.n = std::sin(el);
  return dc;
}

DirectionCosines solar_direction(std::chrono::system_clock::time_point utc_time,
                                 double latitude_deg, double longitude_deg, double height_m) {
  using namespace casacore;
  MPosition site(MVPosition(Quantity(height_m, "m"), Quantity(longitude_deg, "deg"),
                            Quantity(latitude_deg, "deg")), MPosition::WGS84);
  MEpoch epoch(Quantity(to_mjd_utc(utc_time), "d"), MEpoch::UTC);
  MeasFrame frame(epoch, site);
  MDirection target(MDirection::Ref(MDirection::SUN, frame));
  // ENU antenna coordinates use geodetic up, as does Astropy AltAz.
  const auto azel = MDirection::Convert(target, MDirection::Ref(MDirection::AZELGEO, frame))();
  const auto angles = azel.getAngle("rad").getValue();
  const double az = angles[0], el = angles[1];
  return {std::sin(az) * std::cos(el), std::cos(az) * std::cos(el), std::sin(el)};
}
