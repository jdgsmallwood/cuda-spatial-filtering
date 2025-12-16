#pragma once
#include "spatial/logging.hpp"
#include "spatial/pinned_vector.hpp"
#include "spatial/spatial.hpp"

#include <atomic>
#include <chrono>
#include <complex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unistd.h>

// forward declarations

template <typename BeamT, typename ArrivalsT> class BeamWriter;
template <typename T> class VisibilitiesWriter;
template <typename TVal, typename TVec> class EigenWriter;

class Output {
public:
  virtual size_t register_beam_data_block(const int start_seq_num,
                                          const int end_seq_num) = 0;
  virtual size_t register_visibilities_block(const int start_seq_num,
                                             const int end_seq_num,
                                             const int num_missing_packets,
                                             const int num_total_packets) = 0;
  virtual size_t
  register_eigendecomposition_data_block(const int start_seq_num,
                                         const int end_seq_num) = 0;

  virtual void *get_beam_data_landing_pointer(const size_t block_num) = 0;
  virtual void *get_visibilities_landing_pointer(const size_t block_num) = 0;
  virtual void *get_arrivals_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_eigenvalues_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_eigenvectors_data_landing_pointer(const size_t block_num) = 0;

  virtual void register_beam_data_transfer_complete(const size_t block_num) = 0;
  virtual void
  register_visibilities_transfer_complete(const size_t block_num) = 0;
  virtual void register_arrivals_transfer_complete(const size_t block_num) = 0;
  virtual void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) = 0;
};

template <typename T> class SingleHostMemoryOutput : public Output {

public:
  using BeamOutput =
      __half[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET][2];
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  using Visibilities = float[T::NR_CHANNELS][T::NR_BASELINES_UNPADDED]
                            [T::NR_POLARIZATIONS][T::NR_POLARIZATIONS][2];
  using Arrivals =
      bool[T::NR_CHANNELS][T::NR_PACKETS_FOR_CORRELATION][T::NR_FPGA_SOURCES];
  using Eigenvalues = typename T::EigenvalueOutputType;
  using Eigenvectors = typename T::EigenvectorOutputType;
  BeamOutput *beam_data;
  Visibilities *visibilities;
  Arrivals *arrivals;
  Eigenvalues *eigenvalues;
  Eigenvectors *eigenvectors;

  size_t register_beam_data_block(const int start_seq_num,
                                  const int end_seq_num) override {
    return 1;
  };
  size_t register_visibilities_block(const int start_seq_num,
                                     const int end_seq_num,
                                     const int num_missing_packets,
                                     const int num_total_packets) override {
    return 1;
  };

  size_t
  register_eigendecomposition_data_block(const int start_seq_num,
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

  void *get_eigenvalues_data_landing_pointer(const size_t block_num) override {
    return (void *)eigenvalues;
  }

  void *get_eigenvectors_data_landing_pointer(const size_t block_num) override {
    return (void *)eigenvectors;
  }

  void register_beam_data_transfer_complete(const size_t block_num) override {};
  void
  register_visibilities_transfer_complete(const size_t block_num) override {};
  void register_arrivals_transfer_complete(const size_t block_num) override {};
  void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&beam_data, sizeof(BeamOutput)));
    CUDA_CHECK(cudaMallocHost((void **)&visibilities, sizeof(Visibilities)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals, sizeof(Arrivals)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvalues, sizeof(Eigenvalues)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvectors, sizeof(Eigenvectors)));
  };
  ~SingleHostMemoryOutput() {
    cudaFreeHost(beam_data);
    cudaFreeHost(visibilities);
    cudaFreeHost(arrivals);
    cudaFreeHost(eigenvalues);
    cudaFreeHost(eigenvectors);
  };
};

template <typename T> class BufferedOutput : public Output {
public:
  using Eigenvalues = typename T::EigenvalueOutputType;
  using Eigenvectors = typename T::EigenvectorOutputType;
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
    int num_missing_packets;
    int num_total_packets;
    bool transfer_complete;

    VisBlock()
        : start_seq_num(0), end_seq_num(0), num_missing_packets(0),
          num_total_packets(0), transfer_complete(false) {}
  };

  struct EigenBlock {
    Eigenvalues eigenvalues;
    Eigenvectors eigenvectors;
    int start_seq_num;
    int end_seq_num;
    bool transfer_complete;

    EigenBlock() : start_seq_num(0), end_seq_num(0), transfer_complete(false) {}
  };

  BufferedOutput(
      std::unique_ptr<BeamWriter<typename T::BeamOutputType,
                                 typename T::ArrivalsOutputType>>
          beam_writer,
      std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
          vis_writer,
      std::unique_ptr<EigenWriter<typename T::EigenvalueOutputType,
                                  typename T::EigenvectorOutputType>>
          eigen_writer,
      size_t beam_buffer_size, size_t vis_buffer_size, size_t eigen_buffer_size)
      : beam_writer_(std::move(beam_writer)),
        vis_writer_(std::move(vis_writer)),
        eigen_writer_(std::move(eigen_writer)), beam_write_idx_(0),
        beam_read_idx_(0), vis_write_idx_(0), vis_read_idx_(0),
        eigen_write_idx_(0), eigen_read_idx_(0), running_(true) {
    // Allocate ring buffer blocks
    beam_blocks_.resize(beam_buffer_size);
    vis_blocks_.resize(vis_buffer_size);
    eigen_blocks_.resize(eigen_buffer_size);
  }

  ~BufferedOutput() {
    running_ = false;

    beam_writer_->flush();
    vis_writer_->flush();
    eigen_writer_->flush();
  }

  size_t register_beam_data_block(const int start_seq_num,
                                  const int end_seq_num) override {
    size_t block_num = beam_write_idx_;

    auto &block = beam_blocks_[block_num];
    block.start_seq_num = start_seq_num;
    block.end_seq_num = end_seq_num;
    block.beam_transfer_complete = false;
    block.arrival_transfer_complete = false;

    LOG_INFO("Beam write index is now {}", beam_write_idx_);
    if (beam_write_idx_ + 1 == beam_read_idx_) {
      LOG_ERROR("Output beam data ring buffer is full. Read Idx is {} and "
                "Write Idx is {}",
                beam_read_idx_, beam_write_idx_);
      while (beam_write_idx_ + 1 == beam_read_idx_) {
        LOG_INFO("Output beam data ring buffer is still full. Waiting....");
        std::this_thread::sleep_for(std::chrono::microseconds(10));
      }
    }
    beam_write_idx_ = (block_num + 1) % beam_blocks_.size();
    LOG_INFO("Updated Beam write idx is now {}", beam_write_idx_);
    return block_num;
  }

  size_t register_visibilities_block(const int start_seq_num,
                                     const int end_seq_num,
                                     const int num_missing_packets,
                                     const int num_total_packets) override {
    size_t block_num = vis_write_idx_;
    vis_write_idx_ = (block_num + 1) % vis_blocks_.size();

    if (vis_write_idx_ == vis_read_idx_) {
      throw std::runtime_error("Visibilities ring buffer is full");
    }

    auto &block = vis_blocks_[block_num];
    block.start_seq_num = start_seq_num;
    block.end_seq_num = end_seq_num;
    block.num_missing_packets = num_missing_packets;
    block.num_total_packets = num_total_packets;
    block.transfer_complete = false;

    return block_num;
  }

  size_t
  register_eigendecomposition_data_block(const int start_seq_num,
                                         const int end_seq_num) override {
    size_t block_num = eigen_write_idx_;
    eigen_write_idx_ = (block_num + 1) % eigen_blocks_.size();

    if (eigen_write_idx_ == eigen_read_idx_) {
      throw std::runtime_error("Eigendata ring buffer is full");
    }
    auto &block = eigen_blocks_[block_num];
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

  void *get_eigenvectors_data_landing_pointer(const size_t block_num) override {
    return &eigen_blocks_[block_num].eigenvectors;
  }

  void *get_eigenvalues_data_landing_pointer(const size_t block_num) override {
    return &eigen_blocks_[block_num].eigenvalues;
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

  void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) override {
    eigen_blocks_[block_num].transfer_complete = true;
  }

  void writer_loop() {

    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();
    LOG_INFO("Writer loop is running!");
    while (running_) {

      if (has_data_to_write()) {
        LOG_INFO("There is data to write! Writing data...");
        cpu_start = clock::now();
        write_beam_data();
        cpu_end = clock::now();

        LOG_DEBUG("CPU time for writing beam data...: {} us",
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      cpu_end - cpu_start)
                      .count());
        cpu_start = clock::now();
        write_visibilities();
        cpu_end = clock::now();
        LOG_DEBUG("CPU time for writing visibilities...: {} us",
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      cpu_end - cpu_start)
                      .count());

        write_eigendata();
      }
    }
  }

  std::atomic<bool> running_{true};

private:
  bool has_data_to_write() {
    bool has_beam = (beam_read_idx_ != beam_write_idx_) &&
                    beam_blocks_[beam_read_idx_].beam_transfer_complete &&
                    beam_blocks_[beam_read_idx_].arrival_transfer_complete;
    bool has_vis = (vis_read_idx_ != vis_write_idx_) &&
                   vis_blocks_[vis_read_idx_].transfer_complete;

    bool has_eigen = (eigen_read_idx_ != eigen_write_idx_) &&
                     eigen_blocks_[eigen_read_idx_].transfer_complete;
    return has_beam || has_vis || has_eigen;
  }

  void write_beam_data() {
    while (beam_read_idx_ != beam_write_idx_ &&
           beam_blocks_[beam_read_idx_].beam_transfer_complete &&
           beam_blocks_[beam_read_idx_].arrival_transfer_complete && running_) {

      const auto &block = beam_blocks_[beam_read_idx_];

      beam_writer_->write_beam_block(&block.beam_data, &block.arrival_data,
                                     block.start_seq_num, block.end_seq_num);
      auto block_num = beam_read_idx_;
      beam_read_idx_ = (beam_read_idx_ + 1) % beam_blocks_.size();
      LOG_INFO("Output block {} written. Next read index is {}.", block_num,
               beam_read_idx_);
    }
  }

  void write_visibilities() {
    while (vis_read_idx_ != vis_write_idx_ &&
           vis_blocks_[vis_read_idx_].transfer_complete && running_) {

      const auto &block = vis_blocks_[vis_read_idx_];

      vis_writer_->write_visibilities_block(
          &block.data, block.start_seq_num, block.end_seq_num,
          block.num_missing_packets, block.num_total_packets);

      vis_read_idx_ = (vis_read_idx_ + 1) % vis_blocks_.size();
    }
  }

  void write_eigendata() {
    while (eigen_read_idx_ != eigen_write_idx_ &&
           eigen_blocks_[eigen_read_idx_].transfer_complete && running_) {

      const auto &block = eigen_blocks_[eigen_read_idx_];

      eigen_writer_->write_eigendata_block(
          &block.eigenvalues, &block.eigenvectors, block.start_seq_num,
          block.end_seq_num);

      eigen_read_idx_ = (eigen_read_idx_ + 1) % eigen_blocks_.size();
    }
  }
  std::unique_ptr<
      BeamWriter<typename T::BeamOutputType, typename T::ArrivalsOutputType>>
      beam_writer_;
  std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
      vis_writer_;
  std::unique_ptr<EigenWriter<typename T::EigenvalueOutputType,
                              typename T::EigenvectorOutputType>>
      eigen_writer_;

  cuda_util::PinnedVector<BeamBlock> beam_blocks_;
  cuda_util::PinnedVector<VisBlock> vis_blocks_;
  cuda_util::PinnedVector<EigenBlock> eigen_blocks_;

  size_t beam_write_idx_;
  size_t beam_read_idx_;
  size_t vis_write_idx_;
  size_t vis_read_idx_;
  size_t eigen_write_idx_;
  size_t eigen_read_idx_;
};
