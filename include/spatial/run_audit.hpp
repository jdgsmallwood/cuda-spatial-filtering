#pragma once

inline void write_hdf5_run_audit(HighFive::File &file,
                                  const CommonArgs &args, int argc,
                                  char *argv[],
                                  int nr_receivers_per_fpga = 10,
                                  int nr_polarizations = 2) {
  using namespace HighFive;
  json manifest;
  manifest["schema_version"] = 1;
  manifest["command_line"] = json::array();
  for (int i = 0; i < argc; ++i) manifest["command_line"].push_back(argv[i]);

  auto env = [](const char *name) -> json {
    const char *value = std::getenv(name);
    return value ? json(value) : json(nullptr);
  };
  manifest["environment"] = {
      {"SPATIAL_CAPTURE_CPUS", env("SPATIAL_CAPTURE_CPUS")},
      {"SPATIAL_WORKER_CPUS", env("SPATIAL_WORKER_CPUS")},
      {"SPATIAL_NUMA_NODE", env("SPATIAL_NUMA_NODE")}};
  manifest["build"] = {
      {"project_version", SPATIAL_PROJECT_VERSION},
      {"git_commit", SPATIAL_GIT_COMMIT},
      {"build_type", SPATIAL_BUILD_TYPE},
      {"compiler", __VERSION__},
      {"compiled_at", std::string(__DATE__) + " " + __TIME__}};

  auto add_file = [&](const std::string &name, const std::string &path) {
    json value = {{"path", path}, {"present", false}, {"content", ""}};
    if (!path.empty()) {
      std::ifstream input(path, std::ios::binary);
      if (input) {
        std::ostringstream content_stream;
        content_stream << input.rdbuf();
        value["present"] = true;
        value["content"] = content_stream.str();
      }
    }
    manifest["input_files"][name] = std::move(value);
  };
  add_file("config", args.config_filename);
  add_file("stream_antenna_map", args.stream_antenna_map_filename);
  add_file("gains", args.gains_filename);
  add_file("fine_delays", args.fine_delays_filename);
  add_file("fpga_delays", args.fpga_delay_file);
  add_file("signal_eigenvalues", args.nr_signal_eigenvectors_filename);
  add_file("beam_weights", args.beam_weights_filename);
  add_file("targets", args.targets_filename);
  manifest["normalized"] = {
      {"config", args.config}, {"gains", args.gains},
      {"beam_weights", args.beam_weights}, {"targets", args.targets},
      {"selected_fpga_ids", args.fpga_id_vec},
      {"network_interfaces", args.fpga_names},
      {"min_frequency_channel", args.min_freq_channel}};

  const bool authoritative = !args.canonical_recv_perm.empty();
  const int streams_per_fpga = nr_receivers_per_fpga * nr_polarizations;
  std::unordered_map<int, int> antenna_to_canonical;
  for (const auto &[index, antenna] : args.canonical_antenna_mapping)
    if (antenna >= 0) antenna_to_canonical[antenna] = index;

  constexpr size_t forward_columns = 12;
  std::vector<int> forward;
  std::unordered_map<int, std::array<int, 4>> reverse_lookup;
  for (int input_f = 0; input_f < (int)args.fpga_id_vec.size(); ++input_f) {
    for (int stream = 0; stream < streams_per_fpga; ++stream) {
      const int global_stream = input_f * streams_per_fpga + stream;
      const int receiver_slot = stream / nr_polarizations;
      const int receiver_index =
          input_f * nr_receivers_per_fpga + receiver_slot;
      const int raw_pol = stream % nr_polarizations;
      int antenna = -1;
      int canonical_pol = raw_pol;
      if (authoritative) {
        auto ant = args.raw_stream_antenna_mapping.find(global_stream);
        auto pol = args.raw_stream_polarization_mapping.find(global_stream);
        if (ant != args.raw_stream_antenna_mapping.end()) antenna = ant->second;
        if (pol != args.raw_stream_polarization_mapping.end())
          canonical_pol = pol->second;
      } else {
        auto ant = args.antenna_mapping.find(receiver_index);
        if (ant != args.antenna_mapping.end()) antenna = ant->second;
      }
      int canonical_index = receiver_index;
      if (authoritative) {
        auto canonical = antenna_to_canonical.find(antenna);
        canonical_index = canonical != antenna_to_canonical.end()
                              ? canonical->second : -1;
      }
      const int disconnected = antenna < 0;
      const int zeroed = authoritative && disconnected;
      const int fpga_id = args.fpga_id_vec[input_f];
      const std::array<int, forward_columns> row{
          global_stream, input_f, fpga_id, stream, receiver_slot, raw_pol,
          canonical_pol, antenna, canonical_index, disconnected, zeroed,
          authoritative ? 1 : 0};
      forward.insert(forward.end(), row.begin(), row.end());
      if (antenna >= 0 && canonical_index >= 0)
        reverse_lookup[canonical_index * nr_polarizations + canonical_pol] =
            {global_stream, input_f, fpga_id, stream};
      manifest["forward_mapping"].push_back({
          {"global_datastream_id", global_stream}, {"fpga_input_index", input_f},
          {"fpga_id", fpga_id},
          {"network_interface", input_f < (int)args.fpga_names.size()
                                    ? args.fpga_names[input_f] : ""},
          {"fpga_stream_id", stream}, {"receiver_slot", receiver_slot},
          {"raw_polarization", raw_pol},
          {"canonical_polarization", canonical_pol},
          {"antenna_id", antenna},
          {"canonical_receiver_index", canonical_index},
          {"configured_disconnected", disconnected != 0},
          {"zeroed_after_reorder", zeroed != 0}});
    }
  }

  constexpr size_t reverse_columns = 8;
  std::vector<int> reverse;
  const int receiver_count =
      args.fpga_id_vec.size() * nr_receivers_per_fpga;
  const auto &receiver_map = authoritative ? args.canonical_antenna_mapping
                                           : args.antenna_mapping;
  for (int canonical_index = 0; canonical_index < receiver_count;
       ++canonical_index) {
    auto ant = receiver_map.find(canonical_index);
    const int antenna = ant != receiver_map.end() ? ant->second : -1;
    for (int pol = 0; pol < nr_polarizations; ++pol) {
      auto raw = reverse_lookup.find(canonical_index * nr_polarizations + pol);
      const std::array<int, 4> source =
          raw != reverse_lookup.end()
              ? raw->second : std::array<int, 4>{-1, -1, -1, -1};
      const int zeroed = authoritative && antenna < 0;
      const std::array<int, reverse_columns> row{
          canonical_index, antenna, pol, source[0], source[1], source[2],
          source[3], zeroed};
      reverse.insert(reverse.end(), row.begin(), row.end());
      manifest["reverse_mapping"].push_back({
          {"canonical_receiver_index", canonical_index},
          {"antenna_id", antenna}, {"canonical_polarization", pol},
          {"global_datastream_id", source[0]}, {"fpga_input_index", source[1]},
          {"fpga_id", source[2]}, {"fpga_stream_id", source[3]},
          {"zeroed_after_reorder", zeroed != 0}});
    }
  }

  Group audit = file.createGroup("audit");
  const std::string manifest_text = manifest.dump(2);
  audit.createDataSet<std::string>("run_manifest_json",
                                   DataSpace::From(manifest_text))
      .write(manifest_text);
  auto forward_ds = audit.createDataSet<int>(
      "forward_stream_mapping",
      DataSpace({forward.size() / forward_columns, forward_columns}));
  forward_ds.write_raw(forward.data());
  forward_ds.createAttribute<std::string>(
      "columns", "global_datastream_id,fpga_input_index,fpga_id,fpga_stream_id,"
      "receiver_slot,raw_polarization,canonical_polarization,antenna_id,"
      "canonical_receiver_index,configured_disconnected,zeroed_after_reorder,"
      "stream_map_authoritative");
  auto reverse_ds = audit.createDataSet<int>(
      "reverse_canonical_mapping",
      DataSpace({reverse.size() / reverse_columns, reverse_columns}));
  reverse_ds.write_raw(reverse.data());
  reverse_ds.createAttribute<std::string>(
      "columns", "canonical_receiver_index,antenna_id,canonical_polarization,"
      "global_datastream_id,fpga_input_index,fpga_id,fpga_stream_id,"
      "zeroed_after_reorder");
}

