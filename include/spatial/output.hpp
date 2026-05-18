#pragma once
#include "spatial/logging.hpp"
#include "spatial/pinned_vector.hpp"
#include "spatial/spatial.hpp"

#include <atomic>
#include <chrono>
#include <complex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <unistd.h>

// forward declarations

template <typename BeamT, typename ArrivalsT> class BeamWriter;
template <typename T> class VisibilitiesWriter;
template <typename TVal, typename TVec> class EigenWriter;
template <typename T> class FFTWriter;

class Output {
public:
  virtual size_t register_beam_data_block(const size_t start_seq_num,
                                          const size_t end_seq_num) = 0;
  virtual size_t register_visibilities_block(const size_t start_seq_num,
                                             const size_t end_seq_num,
                                             const int num_missing_packets,
                                             const int num_total_packets) = 0;
  virtual size_t
  register_eigendecomposition_data_block(const size_t start_seq_num,
                                         const size_t end_seq_num) = 0;

  virtual size_t register_fft_block(const size_t start_seq_num,
                                    const size_t end_seq_num,
                                    const int channel_idx,
                                    const int pol_idx) = 0;

  virtual void *get_beam_data_landing_pointer(const size_t block_num) = 0;
  virtual void *get_visibilities_landing_pointer(const size_t block_num) = 0;
  virtual void *get_arrivals_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_eigenvalues_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_eigenvectors_data_landing_pointer(const size_t block_num) = 0;
  virtual void *get_fft_landing_pointer(const size_t block_num) = 0;

  virtual void register_beam_data_transfer_complete(const size_t block_num) = 0;
  virtual void
  register_visibilities_transfer_complete(const size_t block_num) = 0;
  virtual void register_arrivals_transfer_complete(const size_t block_num) = 0;
  virtual void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) = 0;
  virtual void register_fft_transfer_complete(const size_t block_num) = 0;
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
  using Arrivals = bool[T::NR_CHANNELS][T::NR_PACKETS_FOR_CORRELATION + 2]
                       [T::NR_FPGA_SOURCES];
  using Eigenvalues = typename T::EigenvalueOutputType;
  using Eigenvectors = typename T::EigenvectorOutputType;
  using FFTOutput = typename T::FFTOutputType;
  BeamOutput *beam_data;
  Visibilities *visibilities;
  Arrivals *arrivals;
  Eigenvalues *eigenvalues;
  Eigenvectors *eigenvectors;
  FFTOutput *fft_output;

  size_t register_beam_data_block(const size_t start_seq_num,
                                  const size_t end_seq_num) override {
    return 1;
  };
  size_t register_visibilities_block(const size_t start_seq_num,
                                     const size_t end_seq_num,
                                     const int num_missing_packets,
                                     const int num_total_packets) override {
    return 1;
  };

  size_t
  register_eigendecomposition_data_block(const size_t start_seq_num,
                                         const size_t end_seq_num) override {
    return 1;
  };
  size_t register_fft_block(const size_t start_seq_num,
                            const size_t end_seq_num, const int channel_idx,
                            const int pol_idx) override {
    return 1;
  }

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
  void *get_fft_landing_pointer(const size_t block_num) override {
    return (void *)fft_output;
  }

  void register_beam_data_transfer_complete(const size_t block_num) override {};
  void
  register_visibilities_transfer_complete(const size_t block_num) override {};
  void register_arrivals_transfer_complete(const size_t block_num) override {};
  void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) override {};
  void register_fft_transfer_complete(const size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&beam_data, sizeof(BeamOutput)));
    CUDA_CHECK(cudaMallocHost((void **)&visibilities, sizeof(Visibilities)));
    CUDA_CHECK(cudaMallocHost((void **)&arrivals, sizeof(Arrivals)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvalues, sizeof(Eigenvalues)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvectors, sizeof(Eigenvectors)));
    CUDA_CHECK(cudaMallocHost((void **)&fft_output, sizeof(FFTOutput)));
  };
  ~SingleHostMemoryOutput() {
    CUDA_CHECK(cudaFreeHost(beam_data));
    CUDA_CHECK(cudaFreeHost(visibilities));
    CUDA_CHECK(cudaFreeHost(arrivals));
    CUDA_CHECK(cudaFreeHost(eigenvalues));
    CUDA_CHECK(cudaFreeHost(eigenvectors));
    CUDA_CHECK(cudaFreeHost(fft_output));
  };
};

template <typename T, typename FFTOutput = typename T::FFTOutputType,
          typename Eigenvalues = typename T::EigenvalueOutputType,
          typename Eigenvectors = typename T::EigenvectorOutputType,
          typename BeamOutputType = typename T::BeamOutputType>
class BufferedOutput : public Output {
public:
  BufferedOutput(
      std::unique_ptr<
          BeamWriter<BeamOutputType, typename T::ArrivalsOutputType>>
          beam_writer,
      std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
          vis_writer,
      std::unique_ptr<EigenWriter<Eigenvalues, Eigenvectors>> eigen_writer,
      std::unique_ptr<FFTWriter<FFTOutput>> fft_writer, size_t beam_buffer_size,
      size_t vis_buffer_size, size_t eigen_buffer_size,
      size_t fft_buffer_size, )
      : beam_writer_(std::move(beam_writer)),
        vis_writer_(std::move(vis_writer)),
        eigen_writer_(std::move(eigen_writer)),
        fft_writer_(std::move(fft_writer)), running_(true) {};

  ~BufferedOutput() {
    running_ = false;

    if (beam_writer_ != nullptr) {
      beam_writer_->flush();
    };
    if (vis_writer_ != nullptr) {
      vis_writer_->flush();
    };
    if (eigen_writer_ != nullptr) {
      eigen_writer_->flush();
    };
    if (fft_writer_ != nullptr) {
      fft_writer_->flush();
    };
  }

  size_t register_beam_data_block(const size_t start_seq_num,
                                  const size_t end_seq_num) override {
    if (beam_writer_ != nullptr) {
      return beam_writer_->register_block(start_seq_num, end_seq_num);
    }
    return std::numeric_limits<size_t>::max();
  }

  size_t register_visibilities_block(const size_t start_seq_num,
                                     const size_t end_seq_num,
                                     const int num_missing_packets,
                                     const int num_total_packets) override {

    if (vis_writer_ != nullptr) {
      return vis_writer_->register_block(
          start_seq_num, end_seq_num, num_missing_packets, num_total_packets);
    }
    return std::numeric_limits<size_t>::max();
  }

  size_t
  register_eigendecomposition_data_block(const size_t start_seq_num,
                                         const size_t end_seq_num) override {

    if (eigen_writer_ != nullptr) {
      return eigen_writer_->register_block(start_seq_num, end_seq_num);
    }
    return std::numeric_limits<size_t>::max();
  }

  size_t register_fft_block(const size_t start_seq_num,
                            const size_t end_seq_num, const int channel_idx,
                            const int pol_idx) override {
    size_t block_num = fft_write_idx_;
    auto &block = fft_blocks_[block_num];
    block.start_seq_num = start_seq_num;
    block.end_seq_num = end_seq_num;
    block.channel_idx = channel_idx;
    block.pol_idx = pol_idx;
    block.transfer_complete = false;

    fft_write_idx_ = (block_num + 1) % fft_blocks_.size();
    LOG_INFO("Registered FFT block with start seq {} and end seq {}",
             start_seq_num, end_seq_num);

    if (fft_write_idx_ == fft_read_idx_) {
      LOG_ERROR("FFT ring buffer is full");
    }
    return block_num;
  }

  void *get_beam_data_landing_pointer(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return nullptr;
    }
    return beam_writer_->get_beam_data_landing_pointer(block_num);
  }

  void *get_visibilities_landing_pointer(const size_t block_num) override {
    if (vis_writer_ == nullptr) {
      return nullptr;
    }
    return vis_writer->get_visibilities_landing_pointer(block_num);
  }

  void *get_arrivals_data_landing_pointer(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return nullptr;
    }
    return beam_writer_->get_arrivals_data_landing_pointer(block_num);
  }

  void *get_eigenvectors_data_landing_pointer(const size_t block_num) override {
    if (eigen_writer_ == nullptr) {
      return nullptr;
    }
    return eigen_writer_->get_eigenvectors_landing_pointer(block_num);
  }

  void *get_eigenvalues_data_landing_pointer(const size_t block_num) override {
    if (eigen_writer_ == nullptr) {
      return nullptr;
    }
    return eigen_writer_->get_eigenvalues_landing_pointer(block_num);
  }

  void *get_fft_landing_pointer(const size_t block_num) override {
    if (fft_writer_ == nullptr) {
      return nullptr;
    }
    return fft_writer_->get_fft_landing_pointer(block_num);
  }

  void register_beam_data_transfer_complete(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return;
    }
    beam_writer_->register_beam_data_transfer_complete(block_num);
  }

  void
  register_visibilities_transfer_complete(const size_t block_num) override {

    if (vis_writer_ == nullptr) {
      return;
    }
    vis_writer_->register_visibilities_transfer_complete(block_num);
  }

  void register_arrivals_transfer_complete(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return;
    }
    beam_writer_->register_arrivals_transfer_complete(block_num);
  }

  void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) override {
    if (eigen_writer_ == nullptr) {
      return;
    }
    eigen_writer_->register_eigendecomposition_transfer_complete(block_num);
  }

  void register_fft_transfer_complete(const size_t block_num) override {
    if (fft_writer_ == nullptr) {
      return;
    }
    fft_writer_->register_fft_transfer_complete(block_num);
  }

  void writer_loop() {

    LOG_INFO("Writer loop is running!");
    while (running_) {

      if (has_data_to_write()) {
        LOG_INFO("There is data to write! Writing data...");
        write_beam_data();
        write_visibilities();
        write_eigendata();
        write_fft_data();
      }
    }
  }

  std::atomic<bool> running_{true};

private:
  bool has_data_to_write() {

    return has_beam || has_vis || has_eigen || has_fft;
  }

  void write_beam_data() {
    LOG_INFO("Beam info....beam_read_idx {}, beam_write_idx {} "
             "transfer_complete: {} arrival transfercomplete {}",
             beam_read_idx_, beam_write_idx_,
             beam_blocks_[beam_read_idx_].beam_transfer_complete,
             beam_blocks_[beam_read_idx_].arrival_transfer_complete);
    while (beam_read_idx_ != beam_write_idx_ &&
           beam_blocks_[beam_read_idx_].beam_transfer_complete &&
           beam_blocks_[beam_read_idx_].arrival_transfer_complete && running_) {

      const auto &block = beam_blocks_[beam_read_idx_];

      if (beam_writer_ != nullptr) {
        beam_writer_->write_beam_block(&block.beam_data, &block.arrival_data,
                                       block.start_seq_num, block.end_seq_num);
      }
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

      if (vis_writer_ != nullptr) {
        vis_writer_->write_visibilities_block(
            &block.data, block.start_seq_num, block.end_seq_num,
            block.num_missing_packets, block.num_total_packets);
      }
      vis_read_idx_ = (vis_read_idx_ + 1) % vis_blocks_.size();
    }
  }

  void write_eigendata() {
    while (eigen_read_idx_ != eigen_write_idx_ &&
           eigen_blocks_[eigen_read_idx_].transfer_complete && running_) {

      const auto &block = eigen_blocks_[eigen_read_idx_];
      if (eigen_writer_ != nullptr) {
        eigen_writer_->write_eigendata_block(
            &block.eigenvalues, &block.eigenvectors, block.start_seq_num,
            block.end_seq_num);
      }
      eigen_read_idx_ = (eigen_read_idx_ + 1) % eigen_blocks_.size();
    }
  }

  void write_fft_data() {

    while (fft_read_idx_ != fft_write_idx_ &&
           fft_blocks_[fft_read_idx_].transfer_complete && running_) {

      const auto &block = fft_blocks_[fft_read_idx_];

      if (fft_writer_ != nullptr) {
        fft_writer_->write_fft_block(&block.fft_output, block.start_seq_num,
                                     block.end_seq_num, block.channel_idx,
                                     block.pol_idx);
      }
      fft_read_idx_ = (fft_read_idx_ + 1) % fft_blocks_.size();
    }
  };

  std::unique_ptr<BeamWriter<BeamOutputType, typename T::ArrivalsOutputType>>
      beam_writer_;
  std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
      vis_writer_;
  std::unique_ptr<EigenWriter<Eigenvalues, Eigenvectors>> eigen_writer_;
  std::unique_ptr<FFTWriter<FFTOutput>> fft_writer_;
};
