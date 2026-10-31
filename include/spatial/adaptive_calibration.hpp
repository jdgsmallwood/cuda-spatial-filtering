#pragma once

// Prepared before opening output files; no invalid receptor silently becomes
// unity.
template <typename T> struct AdaptiveCalibration {
  typename T::FineAntennaGains gains{};
  std::vector<uint8_t> enabled;
  json exclusions = json::array();
};

template <typename T>
AdaptiveCalibration<T> load_adaptive_calibration(const CommonArgs &args,
                                                 bool calibrated, bool coarse,
                                                 int rank) {
  AdaptiveCalibration<T> result;
  const auto &mapping = args.canonical_recv_perm.empty()
                            ? args.antenna_mapping
                            : args.canonical_antenna_mapping;
  if (calibrated) {
    if (!args.gains.contains("weights") ||
        !args.gains.at("weights").is_object())
      throw std::invalid_argument("Calibration requires a weights object");
    if (T::NR_FINE_CHANNELS > 1 &&
        args.gains.value("channel_axis", std::string{}) !=
            "fine_processing_index")
      throw std::invalid_argument(
          "Calibration requires channel_axis=fine_processing_index");
    if (!args.deripple_config_filename.empty())
      validate_deripple_calibration(
          args.gains, load_coarse_pfb_deripple(args.deripple_config_filename,
                                               T::NR_FINE_CHANNELS,
                                               T::NR_FINE_CHANNEL_EDGE_TRIM));
  }
  std::vector<std::string> reasons(T::NR_CHANNELS * 2 * T::NR_RECEIVERS);
  for (size_t c = 0; c < T::NR_CHANNELS; ++c)
    for (size_t p = 0; p < 2; ++p)
      for (size_t r = 0; r < T::NR_RECEIVERS; ++r) {
        const int antenna = mapping.at(r);
        const std::string pol = p == 0 ? "XX" : "YY";
        auto &reason = reasons[(c * 2 + p) * T::NR_RECEIVERS + r];
        if (antenna < 0)
          reason = "disconnected";
        if (std::find(args.beam_excluded_antennas.begin(),
                      args.beam_excluded_antennas.end(),
                      antenna) != args.beam_excluded_antennas.end())
          reason = "explicit exclusion";
        if (calibrated && args.gains.contains("excluded_receptors"))
          for (const auto &entry : args.gains.at("excluded_receptors")) {
            const auto name = entry.at("polarization").get<std::string>();
            if (name != "XX" && name != "YY")
              throw std::invalid_argument(
                  "excluded_receptors polarization must be XX or YY");
            if (entry.at("antenna_id").get<int>() == antenna && name == pol)
              reason = "calibration exclusion";
          }
        std::complex<float> inverse{1, 0};
        if (reason.empty() && calibrated) {
          try {
            const auto &entry =
                args.gains.at("weights")
                    .at(std::to_string(args.min_freq_channel + c))
                    .at(pol)
                    .at(std::to_string(antenna));
            const std::complex<float> measured{entry.at("real").get<float>(),
                                               entry.at("imag").get<float>()};
            const float power = std::norm(measured);
            if (!std::isfinite(power) || power < 1e-12f)
              throw std::invalid_argument("zero or non-finite gain");
            inverse = std::conj(measured) / power;
            if (std::abs(inverse) > 65504.f)
              throw std::invalid_argument("inverse gain exceeds fp16 range");
          } catch (const std::exception &) {
            reason = "missing or invalid gain";
          }
        }
        result.gains[c][p][r] =
            reason.empty() ? inverse : std::complex<float>{0, 0};
      }
  if (coarse)
    for (size_t c = 0; c < T::NR_FPGA_CHANNELS; ++c)
      for (size_t p = 0; p < 2; ++p)
        for (size_t r = 0; r < T::NR_RECEIVERS; ++r) {
          bool excluded = false;
          for (size_t f = 0; f < T::NR_EFFECTIVE_FINE_CHANNELS; ++f)
            excluded |=
                std::norm(result.gains[c * T::NR_EFFECTIVE_FINE_CHANNELS + f][p]
                                      [r]) == 0;
          if (excluded)
            for (size_t f = 0; f < T::NR_EFFECTIVE_FINE_CHANNELS; ++f) {
              const size_t fine = c * T::NR_EFFECTIVE_FINE_CHANNELS + f;
              result.gains[fine][p][r] = {0, 0};
              auto &reason = reasons[(fine * 2 + p) * T::NR_RECEIVERS + r];
              if (reason.empty())
                reason = "shared coarse-filter exclusion";
            }
        }
  for (size_t c = 0; c < T::NR_CHANNELS; ++c)
    for (size_t p = 0; p < 2; ++p) {
      int active = 0;
      for (size_t r = 0; r < T::NR_RECEIVERS; ++r) {
        const auto &reason = reasons[(c * 2 + p) * T::NR_RECEIVERS + r];
        active += reason.empty();
        if (!reason.empty())
          result.exclusions.push_back({{"processing_channel", c},
                                       {"receiver", r},
                                       {"antenna", mapping.at(r)},
                                       {"polarization", p == 0 ? "XX" : "YY"},
                                       {"reason", reason}});
        if (!coarse || c % T::NR_EFFECTIVE_FINE_CHANNELS == 0)
          result.enabled.push_back(reason.empty());
      }
      if (active == 0 || rank >= active)
        throw std::invalid_argument(
            "Too few calibrated inputs for the requested nulling rank");
    }
  return result;
}
