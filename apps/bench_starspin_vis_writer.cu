// Measure the actual HDF5 visibility writer with 28- and 27-bin Starspin
// output layouts. No packet capture or GPU pipeline is involved.
#include "spatial/common.hpp"
#include <chrono>
#include <cstring>
#include <filesystem>
#include <iostream>

template <int ExtraLowerTrim>
using Config = LambdaConfig<40, 4, 64, 40, 2, 10, 256, 1, 40, 32, 56,
                            true, 256, 32, 2, 36, ExtraLowerTrim>;

template <typename T>
void run(int blocks, const std::filesystem::path &directory) {
  const auto file_path = directory /
      ("vis_writer_40_" + std::to_string(T::NR_EFFECTIVE_FINE_CHANNELS) + ".h5");
  constexpr size_t bytes = sizeof(typename T::VisibilitiesOutputType);
  const auto begin = std::chrono::steady_clock::now();
  {
    HighFive::File file(file_path.string(), HighFive::File::Truncate);
    HDF5VisibilitiesWriter<typename T::VisibilitiesOutputType> writer(
        file, 0, 39, nullptr, 100, 0);
    writer.start();
    for (int block = 0; block < blocks; ++block) {
      const size_t slot = writer.register_block(block, block + 1, 0, 1);
      std::memset(writer.get_visibilities_landing_pointer(slot), 0, bytes);
      writer.register_visibilities_transfer_complete(slot);
      writer.notify();
    }
    writer.stop();
    file.flush();
  }
  const double seconds = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - begin).count();
  std::cout << "fine_bins=" << T::NR_EFFECTIVE_FINE_CHANNELS
            << " blocks=" << blocks << " bytes_per_block=" << bytes
            << " seconds=" << seconds << " blocks_per_s=" << blocks / seconds
            << " GB_per_s=" << blocks * double(bytes) / seconds / 1e9
            << " file=" << file_path << std::endl;
}

int main(int argc, char **argv) {
  const int blocks = argc > 1 ? std::atoi(argv[1]) : 96;
  const std::filesystem::path directory = argc > 2 ? argv[2] : "/tmp";
  if (blocks < 1 || blocks > 1000) return 1;
  std::filesystem::create_directories(directory);
  run<Config<0>>(blocks, directory);
  run<Config<1>>(blocks, directory);
}
