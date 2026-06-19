#include "spatial/common.hpp"
#include <chrono>
#include <cstring>

// Fixed representative configs for the LAMBDA instrument -- same three shapes
// as bench_processor so the writer numbers are directly comparable.
//                                    ch  fp   ts   rx  pol rxpp corr  bm  pad  blk       acc
using Cfg1ch1fpga = LambdaConfig<     1,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg8ch1fpga = LambdaConfig<     8,  1,   64,  10,  2,  10, 256,  1,  32,  32, 10000000>;
using Cfg8ch4fpga = LambdaConfig<     8,  4,   64,  40,  2,  10, 256,  1,  64,  32, 10000000>;

struct WriterBenchResult {
  size_t nr_channels;
  size_t nr_fpga_sources;
  size_t nr_receivers;
  // Visibilities writer
  size_t vis_blocks;
  size_t vis_bytes_per_block;
  double vis_elapsed;
  double vis_blocks_per_sec;
  double vis_gb_per_sec;
  // Eigendata writer
  size_t eigen_blocks;
  size_t eigen_bytes_per_block;
  double eigen_elapsed;
  double eigen_blocks_per_sec;
  double eigen_gb_per_sec;
};

template <typename Config>
WriterBenchResult run_writers_bench(double duration_s, int num_blocks,
                                    const std::string &output_dir) {
  const std::string tag = std::to_string(Config::NR_CHANNELS) + "ch_" +
                          std::to_string(Config::NR_FPGA_SOURCES) + "fpga";

  WriterBenchResult r;
  r.nr_channels = Config::NR_CHANNELS;
  r.nr_fpga_sources = Config::NR_FPGA_SOURCES;
  r.nr_receivers = Config::NR_RECEIVERS;

  // ---- Visibilities writer ----
  {
    const std::string filename =
        output_dir + "/bench_writers_vis_" + tag + ".h5";
    HighFive::File file(filename, HighFive::File::Truncate);
    HDF5VisibilitiesWriter<typename Config::VisibilitiesOutputType> writer(
        file, 0, static_cast<int>(Config::NR_CHANNELS) - 1, nullptr,
        num_blocks);
    writer.start();

    constexpr size_t bytes_per_block = sizeof(typename Config::VisibilitiesOutputType);
    size_t blocks_written = 0;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
               .count() < duration_s) {
      size_t idx = writer.register_block(blocks_written, blocks_written + 1, 0,
                                         Config::NR_PACKETS_FOR_CORRELATION);
      void *ptr = writer.get_visibilities_landing_pointer(idx);
      std::memset(ptr, 0, bytes_per_block);
      writer.register_visibilities_transfer_complete(idx);
      writer.notify();
      blocks_written++;
    }
    const auto end = std::chrono::steady_clock::now();
    writer.stop();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    r.vis_blocks = blocks_written;
    r.vis_bytes_per_block = bytes_per_block;
    r.vis_elapsed = elapsed;
    r.vis_blocks_per_sec = blocks_written / elapsed;
    r.vis_gb_per_sec =
        static_cast<double>(bytes_per_block) * blocks_written / elapsed / 1e9;
  }

  // ---- Eigendata writer ----
  {
    const std::string filename =
        output_dir + "/bench_writers_eigen_" + tag + ".h5";
    HighFive::File file(filename, HighFive::File::Truncate);
    HDF5EigenWriter<typename Config::EigenvalueOutputType,
                    typename Config::EigenvectorOutputType>
        writer(file, 0, num_blocks);
    writer.start();

    constexpr size_t bytes_per_block =
        sizeof(typename Config::EigenvalueOutputType) +
        sizeof(typename Config::EigenvectorOutputType);
    size_t blocks_written = 0;
    const auto start = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                         start)
               .count() < duration_s) {
      size_t idx = writer.register_block(blocks_written, blocks_written + 1);
      void *val_ptr = writer.get_eigenvalues_landing_pointer(idx);
      void *vec_ptr = writer.get_eigenvectors_landing_pointer(idx);
      std::memset(val_ptr, 0, sizeof(typename Config::EigenvalueOutputType));
      std::memset(vec_ptr, 0, sizeof(typename Config::EigenvectorOutputType));
      writer.register_eigendecomposition_transfer_complete(idx);
      writer.notify();
      blocks_written++;
    }
    const auto end = std::chrono::steady_clock::now();
    writer.stop();
    const double elapsed = std::chrono::duration<double>(end - start).count();

    r.eigen_blocks = blocks_written;
    r.eigen_bytes_per_block = bytes_per_block;
    r.eigen_elapsed = elapsed;
    r.eigen_blocks_per_sec = blocks_written / elapsed;
    r.eigen_gb_per_sec =
        static_cast<double>(bytes_per_block) * blocks_written / elapsed / 1e9;
  }

  return r;
}

static void print_result(const WriterBenchResult &r) {
  std::printf(
      "[Vis Writer ch=%zu fpga=%zu rx=%zu] "
      "blocks=%zu elapsed=%.3f bytes_per_block=%zu "
      "blocks/sec=%.4f GB/sec=%.6f\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers, r.vis_blocks,
      r.vis_elapsed, r.vis_bytes_per_block, r.vis_blocks_per_sec,
      r.vis_gb_per_sec);
  std::printf(
      "[Eigen Writer ch=%zu fpga=%zu rx=%zu] "
      "blocks=%zu elapsed=%.3f bytes_per_block=%zu "
      "blocks/sec=%.4f GB/sec=%.6f\n",
      r.nr_channels, r.nr_fpga_sources, r.nr_receivers, r.eigen_blocks,
      r.eigen_elapsed, r.eigen_bytes_per_block, r.eigen_blocks_per_sec,
      r.eigen_gb_per_sec);
  std::fflush(stdout);
}

int main(int argc, char *argv[]) {
  argparse::ArgumentParser program("bench_writers");
  program.add_argument("--duration")
      .help("Duration in seconds to run each writer benchmark per config")
      .default_value(10.0)
      .scan<'g', double>();
  program.add_argument("--num-blocks")
      .help("Writer ring-buffer size / HDF5 flush batch size")
      .default_value(100)
      .scan<'i', int>();
  program.add_argument("-o", "--output-dir")
      .help("Directory to write scratch HDF5 output files")
      .default_value(std::string("."));

  try {
    program.parse_args(argc, argv);
  } catch (const std::exception &err) {
    std::cerr << err.what() << std::endl;
    std::cerr << program;
    return 1;
  }

  const double duration_s = program.get<double>("--duration");
  const int num_blocks = program.get<int>("--num-blocks");
  const std::string output_dir = program.get<std::string>("-o");

  std::cout << "bench_writers: 3 configs x " << duration_s << "s each"
            << " num_blocks=" << num_blocks << std::endl;

  print_result(run_writers_bench<Cfg1ch1fpga>(duration_s, num_blocks, output_dir));
  print_result(run_writers_bench<Cfg8ch1fpga>(duration_s, num_blocks, output_dir));
  print_result(run_writers_bench<Cfg8ch4fpga>(duration_s, num_blocks, output_dir));

  return 0;
}
