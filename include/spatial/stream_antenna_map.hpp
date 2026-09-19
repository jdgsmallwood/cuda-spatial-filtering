#pragma once

#include <algorithm>
#include <fstream>
#include <map>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

// Loads stream_antenna_map.json: maps each (FPGA, stream) pair to a physical
// (antenna_id, canonical_polarization).  "stream" is the flat index within an FPGA:
//   stream k  =  receiver_slot (k / NR_POL),  pol_slot (k % NR_POL)
// "polarization": 0 = X, 1 = Y — the canonical label for the signal at that hw sample position.
// antenna_id == -1 means the stream is disconnected.
struct StreamAntennaMap {
  struct StreamEntry { int antenna_id; int polarization; };
  // entries[fpga_id][stream_flat] = StreamEntry
  std::unordered_map<int, std::unordered_map<int, StreamEntry>> entries;

  static StreamAntennaMap load(const std::string &path) {
    StreamAntennaMap result;
    std::ifstream f(path);
    if (!f.is_open())
      throw std::runtime_error("Cannot open stream antenna map: " + path);
    json j = json::parse(f);
    for (const auto &[fpga_str, fpga_data] : j.at("fpgas").items()) {
      int fpga_id = std::stoi(fpga_str);
      for (const auto &s : fpga_data.at("streams")) {
        int stream       = s.at("stream");
        int antenna_id   = s.at("antenna_id");
        int polarization = s.at("polarization");
        result.entries[fpga_id][stream] = {antenna_id, polarization};
      }
    }
    return result;
  }

  // Returns {recv_perm, pol_perm}, each of length
  //   nr_fpgas * nr_recv_per_fpga * nr_polarizations.
  //
  // Index: (canonical_recv_flat * nr_polarizations + canonical_pol), where
  //   canonical_recv_flat = out_f * nr_recv_per_fpga + out_n.
  //
  // recv_perm[idx] = hw flat receiver (fpga * nr_recv_per_fpga + stream / nr_pol),
  //                  or -1 = unused / disconnected slot (kernel zeroes output).
  // pol_perm [idx] = hw pol slot (stream % nr_pol) within that receiver.
  //
  // Canonical order: ascending antenna_id; unused/disconnected slots at the end.
  //
  // Validation: every connected antenna_id must appear with all nr_polarizations values
  // (0 .. nr_polarizations-1).  Throws std::runtime_error if any polarization is missing.
  std::pair<std::vector<int>, std::vector<int>>
  build_permutation(const std::vector<int> &fpga_ids, int nr_recv_per_fpga,
                    int nr_polarizations) const {
    const int nr_fpgas = static_cast<int>(fpga_ids.size());
    int total = nr_fpgas * nr_recv_per_fpga * nr_polarizations;

    // Collect connected stream entries per antenna_id and canonical_pol.
    // ant_map[antenna_id][canonical_pol] = {hw_flat_recv, hw_pol_slot}
    std::map<int, std::map<int, std::pair<int, int>>> ant_map;
    for (int f = 0; f < nr_fpgas; ++f) {
      const int fpga_id = fpga_ids[f];
      for (int s = 0; s < nr_recv_per_fpga * nr_polarizations; ++s) {
        if (!entries.count(fpga_id) || !entries.at(fpga_id).count(s)) continue;
        const auto &e = entries.at(fpga_id).at(s);
        if (e.antenna_id < 0) continue;
        int hw_recv = f * nr_recv_per_fpga + s / nr_polarizations;
        int hw_pol  = s % nr_polarizations;
        ant_map[e.antenna_id][e.polarization] = {hw_recv, hw_pol};
      }
    }

    // Validate: each connected antenna must have entries for all nr_polarizations values.
    for (const auto &[antenna_id, pol_map] : ant_map) {
      for (int p = 0; p < nr_polarizations; ++p) {
        if (!pol_map.count(p))
          throw std::runtime_error(
              "StreamAntennaMap: antenna " + std::to_string(antenna_id) +
              " is missing polarization=" + std::to_string(p) + " entry");
      }
    }

    // Sort connected antennas by antenna_id for canonical ordering.
    std::vector<int> sorted_ants;
    for (const auto &[aid, _] : ant_map) sorted_ants.push_back(aid);
    std::sort(sorted_ants.begin(), sorted_ants.end());

    std::vector<int> recv_perm(total, -1), pol_perm(total, 0);
    for (int c = 0; c < (int)sorted_ants.size(); ++c) {
      const auto &pol_map = ant_map.at(sorted_ants[c]);
      for (int p = 0; p < nr_polarizations; ++p) {
        int idx = c * nr_polarizations + p;
        auto [hw_recv, hw_pol] = pol_map.at(p);
        recv_perm[idx] = hw_recv;
        pol_perm[idx]  = hw_pol;
      }
    }
    return {recv_perm, pol_perm};
  }

  // Build canonical receiver identities directly from this map in the same
  // ascending antenna-ID order used by build_permutation().
  std::unordered_map<int, int>
  build_canonical_antenna_mapping(const std::vector<int> &fpga_ids,
                                  int nr_recv_per_fpga,
                                  int nr_polarizations) const {
    std::set<int> connected_antennas;
    for (int fpga_id : fpga_ids) {
      auto fpga_it = entries.find(fpga_id);
      if (fpga_it == entries.end()) continue;
      for (const auto &[stream, entry] : fpga_it->second) {
        if (stream >= 0 && stream < nr_recv_per_fpga * nr_polarizations &&
            entry.antenna_id >= 0)
          connected_antennas.insert(entry.antenna_id);
      }
    }

    std::unordered_map<int, int> result;
    const int nr_canonical = static_cast<int>(fpga_ids.size()) * nr_recv_per_fpga;
    int c = 0;
    for (int antenna_id : connected_antennas) result[c++] = antenna_id;
    while (c < nr_canonical) result[c++] = -1;
    return result;
  }
};
