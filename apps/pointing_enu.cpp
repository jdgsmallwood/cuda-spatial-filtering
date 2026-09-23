// Print the exact casacore direction used by the beam-steering pipeline.
#include "spatial/pointing.hpp"

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc != 7) {
    std::cerr << "usage: pointing_enu UTC_ISO RA_DEG DEC_DEG LAT_DEG LON_DEG HEIGHT_M\n";
    return 2;
  }
  std::tm parsed{};
  std::istringstream in(argv[1]);
  in >> std::get_time(&parsed, "%Y-%m-%dT%H:%M:%SZ");
  if (in.fail() || in.peek() != std::char_traits<char>::eof())
    throw std::invalid_argument("UTC must be YYYY-MM-DDTHH:MM:SSZ");
  const auto instant = std::chrono::system_clock::from_time_t(timegm(&parsed));
  const auto dc = topocentric_direction(
      std::stod(argv[2]), std::stod(argv[3]), instant,
      std::stod(argv[4]), std::stod(argv[5]), std::stod(argv[6]));
  std::cout << std::setprecision(17) << dc.l << ' ' << dc.m << ' ' << dc.n << '\n';
}
