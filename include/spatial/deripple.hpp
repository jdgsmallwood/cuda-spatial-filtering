#pragma once

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

// This file deliberately contains no production PFB coefficients. The table is
// supplied at runtime, after its firmware provenance has been checked.
struct CoarsePfbDeripple {
  std::vector<float> voltage_gains;
  std::string response_id;
};

// parse_common_args configures this before any production pipeline is built.
// FineChannelizer owns the device table, so a captured CUDA graph always sees
// a stable pointer and no process-global GPU allocation is leaked.
inline std::string &coarse_pfb_deripple_config_path() {
  static std::string path;
  return path;
}

inline CoarsePfbDeripple load_coarse_pfb_deripple(
    const std::string &filename, int fine_channels, int edge_trim) {
  std::ifstream file(filename);
  if (!file)
    throw std::runtime_error("Cannot open de-ripple config: " + filename);
  nlohmann::json config;
  try {
    file >> config;
    if (config.at("schema_version").get<int>() != 1 ||
        config.at("kind").get<std::string>() !=
            "coarse_pfb_amplitude_deripple" ||
        config.at("fine_channels").get<int>() != fine_channels ||
        config.at("edge_trim").get<int>() != edge_trim ||
        config.at("normalization").get<std::string>() !=
            "coarse_bin_centre")
      throw std::runtime_error("schema, layout or normalization mismatch");
    CoarsePfbDeripple result;
    result.response_id = config.at("response_id").get<std::string>();
    if (result.response_id.empty() || result.response_id.size() > 64 ||
        result.response_id.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.") !=
            std::string::npos)
      throw std::runtime_error("response_id must be 1-64 safe ASCII characters");
    const auto &gains = config.at("voltage_gains");
    if (!gains.is_array() ||
        gains.size() != static_cast<size_t>(fine_channels - 2 * edge_trim))
      throw std::runtime_error("voltage_gains length mismatch");
    for (const auto &entry : gains) {
      if (!entry.is_number() || entry.is_boolean())
        throw std::runtime_error("voltage_gains entries must be numbers");
      const float gain = entry.get<float>();
      if (!std::isfinite(gain) || gain <= 0.0f || gain > 1.1f)
        throw std::runtime_error("voltage_gains must be finite and in (0, 1.1]");
      result.voltage_gains.push_back(gain);
    }
    return result;
  } catch (const std::exception &error) {
    throw std::runtime_error("Invalid de-ripple config " + filename + ": " +
                             error.what());
  }
}

inline void validate_deripple_calibration(
    const nlohmann::json &measured_gains, const CoarsePfbDeripple &deripple) {
  try {
    const auto &provenance = measured_gains.at("provenance");
    if (provenance.at("deripple_response_id").get<std::string>() !=
            deripple.response_id ||
        provenance.at("deripple_export_state").get<std::string>() !=
            "measured_from_derippled_visibilities" ||
        provenance.at("deripple_voltage_gains").get<std::vector<float>>() !=
            deripple.voltage_gains)
      throw std::runtime_error("response ID, voltage table or export state mismatch");
  } catch (const std::exception &error) {
    throw std::runtime_error(
        "De-ripple and fine beam gains are incompatible; re-export gains "
        "with --deripple-config: " + std::string(error.what()));
  }
}
