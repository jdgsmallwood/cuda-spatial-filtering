#include "spatial/logging.hpp"
#include "spatial/spatial.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unistd.h>

// forward declarations

template <typename BeamT, typename ArrivalsT> class BeamWriter;
template <typename T> class VisibilitiesWriter;

class Output {
public:
  virtual size_t register_beam_data_block(const int start_seq_num,
                                          const int end_seq_num) = 0;
  virtual size_t register_visibilities_block(const int start_seq_num,
                                             const int end_seq_num) = 0;

  virtual void *get_beam_data_landing_pointer(const size_t block_num) = 0;
  virtual void *get_visibilities_landing_pointer(const size_t block_num) = 0;
  virtual void *get_arrivals_data_landing_pointer(const size_t block_num) = 0;

  virtual void register_beam_data_transfer_complete(const size_t block_num) = 0;
  virtual void
  register_visibilities_transfer_complete(const size_t block_num) = 0;
  virtual void register_arrivals_transfer_complete(const size_t block_num) = 0;
};

template <typename T> class SingleHostMemoryOutput : public Output {

public:
  using BeamOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET][2];
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  using Visibilities = float[T::NR_CHANNELS][NR_BASELINES][T::NR_POLARIZATIONS]
                            [T::NR_POLARIZATIONS][2];
  using Arrivals =
      bool[T::NR_CHANNELS][T::NR_PACKETS_FOR_CORRELATION][T::NR_FPGA_SOURCES];
  BeamOutput *beam_data;
  Visibilities *visibilities;
  Arrivals *arrivals;

  size_t register_beam_data_block(const int start_seq_num,
                                  const int end_seq_num) override {
    return 1;
  };
  size_t register_visibilities_block(const int start_seq_num,
                                     const int end_seq_num) override {
    return 1;
  };

  void *get_beam_data_landing_pointer(const size_t block_num) override {
    return (void *)beam_data;
  };

  void *get_visibilities_landing_pointer(const size_t block_num) override {
    return (void *)visibilities;
  }

  void *get_arrivals_data_landing_pointer(const size_t block_num) override {
    return (void *)arrivals;
  }

  void register_beam_data_transfer_complete(const size_t block_num) override {};
  void
  register_visibilities_transfer_complete(const size_t block_num) override {};
  void register_arrivals_transfer_complete(const size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&beam_data, sizeof(BeamOutput)));
    CUDA_CHECK(cudaMallocHost((void **)&visibilities, sizeof(Visibilities)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals, sizeof(Arrivals)));
  };
  ~SingleHostMemoryOutput() {
    cudaFreeHost(beam_data);
    cudaFreeHost(visibilities);
    cudaFreeHost(arrivals);
  };
};

template <typename T> class BufferedOutput : public Output {
public:
  struct BeamBlock {
    typename T::BeamOutputType beam_data;
    typename T::ArrivalsOutputType arrival_data;
    int start_seq_num;
    int end_seq_num;
    bool beam_transfer_complete;
    bool arrival_transfer_complete;

    BeamBlock()
        : start_seq_num(0), end_seq_num(0), beam_transfer_complete(false),
          arrival_transfer_complete(false) {}
  };

  struct VisBlock {
    typename T::VisibilitiesOutputType data;
    int start_seq_num;
    int end_seq_num;
    bool transfer_complete;

    VisBlock() : start_seq_num(0), end_seq_num(0), transfer_complete(false) {}
  };

  BufferedOutput(
      std::unique_ptr<BeamWriter<typename T::BeamOutputType,
                                 typename T::ArrivalsOutputType>>
          beam_writer,
      std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
          vis_writer,
      size_t beam_buffer_size, size_t vis_buffer_size)
      : beam_writer_(std::move(beam_writer)),
        vis_writer_(std::move(vis_writer)), beam_write_idx_(0),
        beam_read_idx_(0), vis_write_idx_(0), vis_read_idx_(0), running_(true) {
    // Allocate ring buffer blocks
    beam_blocks_.resize(beam_buffer_size);
    vis_blocks_.resize(vis_buffer_size);

    // Start writer thread
    writer_thread_ = std::thread(&BufferedOutput::writer_loop, this);
  }

  ~BufferedOutput() {
    running_ = false;

    if (writer_thread_.joinable()) {
      writer_thread_.join();
    }
    beam_writer_->flush();
    vis_writer_->flush();
  }

  size_t register_beam_data_block(const int start_seq_num,
                                  const int end_seq_num) override {
    size_t block_num = beam_write_idx_;
    beam_write_idx_ = (block_num + 1) % beam_blocks_.size();

    if (beam_write_idx_ == beam_read_idx_) {
      throw std::runtime_error("Beam data ring buffer is full");
    }

    auto &block = beam_blocks_[block_num];
    block.start_seq_num = start_seq_num;
    block.end_seq_num = end_seq_num;
    block.beam_transfer_complete = false;
    block.arrival_transfer_complete = false;

    return block_num;
  }

  size_t register_visibilities_block(const int start_seq_num,
                                     const int end_seq_num) override {
    size_t block_num = vis_write_idx_;
    vis_write_idx_ = (block_num + 1) % vis_blocks_.size();

    if (vis_write_idx_ == vis_read_idx_) {
      throw std::runtime_error("Visibilities ring buffer is full");
    }

    auto &block = vis_blocks_[block_num];
    block.start_seq_num = start_seq_num;
    block.end_seq_num = end_seq_num;
    block.transfer_complete = false;

    return block_num;
  }

  void *get_beam_data_landing_pointer(const size_t block_num) override {
    return &beam_blocks_[block_num].beam_data;
  }

  void *get_visibilities_landing_pointer(const size_t block_num) override {
    return &vis_blocks_[block_num].data;
  }

  void *get_arrivals_data_landing_pointer(const size_t block_num) override {
    return &beam_blocks_[block_num].arrival_data;
  }

  void register_beam_data_transfer_complete(const size_t block_num) override {
    beam_blocks_[block_num].beam_transfer_complete = true;
  }

  void
  register_visibilities_transfer_complete(const size_t block_num) override {
    vis_blocks_[block_num].transfer_complete = true;
  }

  void register_arrivals_transfer_complete(const size_t block_num) override {
    beam_blocks_[block_num].arrival_transfer_complete = true;
  }

private:
  void writer_loop() {
    LOG_INFO("Writer loop is running!");
    while (running_) {

      if (has_data_to_write()) {
        LOG_INFO("There is data to write! Writing data...");
        write_beam_data();
        write_visibilities();
      }
    }

    // Flush remaining data
    while (beam_read_idx_ != beam_write_idx_ ||
           vis_read_idx_ != vis_write_idx_) {
      write_beam_data();
      write_visibilities();
    }
  }

  bool has_data_to_write() {
    bool has_beam = (beam_read_idx_ != beam_write_idx_) &&
                    beam_blocks_[beam_read_idx_].beam_transfer_complete &&
                    beam_blocks_[beam_read_idx_].arrival_transfer_complete;
    bool has_vis = (vis_read_idx_ != vis_write_idx_) &&
                   vis_blocks_[vis_read_idx_].transfer_complete;

    return has_beam || has_vis;
  }

  void write_beam_data() {
    while (beam_read_idx_ != beam_write_idx_ &&
           beam_blocks_[beam_read_idx_].beam_transfer_complete &&
           beam_blocks_[beam_read_idx_].arrival_transfer_complete) {

      const auto &block = beam_blocks_[beam_read_idx_];

      beam_writer_->write_beam_block(&block.beam_data, &block.arrival_data,
                                     block.start_seq_num, block.end_seq_num);

      beam_read_idx_ = (beam_read_idx_ + 1) % beam_blocks_.size();
    }
  }

  void write_visibilities() {
    while (vis_read_idx_ != vis_write_idx_ &&
           vis_blocks_[vis_read_idx_].transfer_complete) {

      const auto &block = vis_blocks_[vis_read_idx_];

      vis_writer_->write_visibilities_block(&block.data, block.start_seq_num,
                                            block.end_seq_num);

      vis_read_idx_ = (vis_read_idx_ + 1) % vis_blocks_.size();
    }
  }

  std::unique_ptr<
      BeamWriter<typename T::BeamOutputType, typename T::ArrivalsOutputType>>
      beam_writer_;
  std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
      vis_writer_;

  std::vector<BeamBlock> beam_blocks_;
  std::vector<VisBlock> vis_blocks_;

  size_t beam_write_idx_;
  size_t beam_read_idx_;
  size_t vis_write_idx_;
  size_t vis_read_idx_;

  std::thread writer_thread_;
  bool running_;
};
