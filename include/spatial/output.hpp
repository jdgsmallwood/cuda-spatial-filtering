#pragma once
#include "spatial/logging.hpp"
#include "spatial/pinned_vector.hpp"
#include "spatial/spatial.hpp"

#include <atomic>
#include <chrono>
#include <complex.h>
#include <cstdint>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>

// forward declarations

template <typename BeamT, typename ArrivalsT, typename TCount>
class BeamWriter;
template <typename T> class VisibilitiesWriter;
template <typename TVal, typename TVec, typename TCount> class EigenWriter;
template <typename T> class FFTWriter;

class Output {
public:
  // Call before ingest can release and clear the input packet's arrival flags.
  size_t register_beam_data_block_with_arrivals(FinalPacketData &packet) {
    const size_t block = register_beam_data_block(packet.start_seq_id, packet.end_seq_id);
    if (block != std::numeric_limits<size_t>::max()) {
      std::memcpy(get_arrivals_data_landing_pointer(block), packet.get_arrivals_ptr(),
                  packet.get_arrivals_size());
      register_arrivals_transfer_complete(block);
    }
    return block;
  }

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
                                    const size_t end_seq_num) = 0;

  virtual void *get_beam_data_landing_pointer(const size_t block_num) = 0;
  virtual void *get_visibilities_landing_pointer(const size_t block_num) = 0;
  virtual void *get_arrivals_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_eigenvalues_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_eigenvectors_data_landing_pointer(const size_t block_num) = 0;
  virtual void *
  get_nulled_eigenmode_counts_landing_pointer(const size_t block_num) = 0;
  virtual void *get_beam_counts_landing_pointer(const size_t block_num) = 0;
  virtual void *get_fft_landing_pointer(const size_t block_num) = 0;

  virtual void register_beam_data_transfer_complete(const size_t block_num) = 0;
  virtual void
  register_visibilities_transfer_complete(const size_t block_num) = 0;
  virtual void register_arrivals_transfer_complete(const size_t block_num) = 0;
  virtual void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) = 0;
  virtual void
  register_beam_counts_transfer_complete(const size_t block_num) = 0;
  virtual void register_fft_transfer_complete(const size_t block_num) = 0;
};

template <typename T> class SingleHostMemoryOutput : public Output {

public:
  // Time axis must match T::NR_TIME_STEPS_PER_FINE_CHANNEL (packet_formats.hpp), not the raw
  // T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET -- the pipeline's beamformer
  // output shrinks to the per-fine-channel length once channelization is active, and this
  // landing buffer must match what execute_pipeline's cudaMemcpyAsync actually transfers.
  using BeamOutput =
      __half[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
            [T::NR_TIME_STEPS_PER_FINE_CHANNEL][2];
  static constexpr int NR_BASELINES =
      T::NR_PADDED_RECEIVERS * (T::NR_PADDED_RECEIVERS + 1) / 2;
  using Visibilities = float[T::NR_CHANNELS][T::NR_BASELINES_UNPADDED]
                            [T::NR_POLARIZATIONS][T::NR_POLARIZATIONS][2];
  // Pre-channelization: arrivals are packet-level (one bit per raw FPGA/coarse channel), matching
  // T::ArrivalsOutputType / packet_data->get_arrivals_size(), not the widened T::NR_CHANNELS.
  using Arrivals = bool[T::NR_FPGA_CHANNELS][T::NR_PACKETS_FOR_CORRELATION + 2]
                       [T::NR_FPGA_SOURCES];
  using Eigenvalues = typename T::EigenvalueOutputType;
  using Eigenvectors = typename T::EigenvectorOutputType;
  using EigenmodeCounts = int32_t[T::NR_CHANNELS][T::NR_POLARIZATIONS];
  using FFTOutput = typename T::FFTOutputType;
  BeamOutput *beam_data;
  Visibilities *visibilities;
#ifdef SPATIAL_THP_OUTPUT_BUFFERS
  void *visibilities_mapping = nullptr;
  size_t visibilities_mapping_bytes = 0;
#endif
  Arrivals *arrivals;
  Eigenvalues *eigenvalues;
  Eigenvectors *eigenvectors;
  EigenmodeCounts *nulled_eigenmode_counts;
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
                            const size_t end_seq_num) override {
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
  void *
  get_nulled_eigenmode_counts_landing_pointer(const size_t block_num) override {
    return (void *)nulled_eigenmode_counts;
  }
  void *get_beam_counts_landing_pointer(const size_t block_num) override {
    return (void *)nulled_eigenmode_counts;
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
  void
  register_beam_counts_transfer_complete(const size_t block_num) override {};
  void register_fft_transfer_complete(const size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&beam_data, sizeof(BeamOutput)));
#ifdef SPATIAL_THP_OUTPUT_BUFFERS
    // This output receives the integrated visibility D2H copy. Match the
    // GPU-local transparent-hugepage input layout, with a pinned fallback.
    cpu_set_t previous_affinity;
    const bool have_affinity =
        sched_getaffinity(0, sizeof(previous_affinity), &previous_affinity) == 0;
    bool restore_affinity = false;
#ifdef SPATIAL_THP_INPUT_BUFFERS
    if (have_affinity) {
      const int cpu = T::PacketFinalDataType::input_buffer_cpu(previous_affinity);
      if (cpu >= 0) {
        cpu_set_t target;
        CPU_ZERO(&target);
        CPU_SET(cpu, &target);
        restore_affinity =
            sched_setaffinity(0, sizeof(target), &target) == 0;
      }
    }
#endif
    constexpr size_t alignment = 2 * 1024 * 1024;
    visibilities_mapping_bytes = sizeof(Visibilities) + alignment;
    visibilities_mapping = mmap(nullptr, visibilities_mapping_bytes,
                                PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    bool registered = false;
    if (visibilities_mapping != MAP_FAILED) {
      const uintptr_t aligned =
          (reinterpret_cast<uintptr_t>(visibilities_mapping) + alignment - 1) &
          ~(uintptr_t(alignment) - 1);
      visibilities = reinterpret_cast<Visibilities *>(aligned);
      if (madvise(visibilities, sizeof(Visibilities), MADV_HUGEPAGE) == 0) {
        std::memset(visibilities, 0, sizeof(Visibilities));
        registered = cudaHostRegister(visibilities, sizeof(Visibilities),
                                      cudaHostRegisterDefault) == cudaSuccess;
      }
      if (!registered) {
        cudaGetLastError();
        munmap(visibilities_mapping, visibilities_mapping_bytes);
      }
    }
    if (!registered) {
      visibilities_mapping = nullptr;
      visibilities_mapping_bytes = 0;
      CUDA_CHECK(cudaMallocHost((void **)&visibilities, sizeof(Visibilities)));
    }
    if (restore_affinity)
      sched_setaffinity(0, sizeof(previous_affinity), &previous_affinity);
#else
    CUDA_CHECK(cudaMallocHost((void **)&visibilities, sizeof(Visibilities)));
#endif
    CUDA_CHECK(cudaMallocHost((void **)&arrivals, sizeof(Arrivals)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvalues, sizeof(Eigenvalues)));
    CUDA_CHECK(cudaMallocHost((void **)&eigenvectors, sizeof(Eigenvectors)));
    CUDA_CHECK(cudaMallocHost((void **)&nulled_eigenmode_counts,
                              sizeof(EigenmodeCounts)));
    CUDA_CHECK(cudaMallocHost((void **)&fft_output, sizeof(FFTOutput)));
  };
  ~SingleHostMemoryOutput() {
    CUDA_CHECK(cudaFreeHost(beam_data));
#ifdef SPATIAL_THP_OUTPUT_BUFFERS
    if (visibilities_mapping) {
      CUDA_CHECK(cudaHostUnregister(visibilities));
      munmap(visibilities_mapping, visibilities_mapping_bytes);
    } else {
      CUDA_CHECK(cudaFreeHost(visibilities));
    }
#else
    CUDA_CHECK(cudaFreeHost(visibilities));
#endif
    CUDA_CHECK(cudaFreeHost(arrivals));
    CUDA_CHECK(cudaFreeHost(eigenvalues));
    CUDA_CHECK(cudaFreeHost(eigenvectors));
    CUDA_CHECK(cudaFreeHost(nulled_eigenmode_counts));
    CUDA_CHECK(cudaFreeHost(fft_output));
  };
};

template <typename T, typename FFTOutput = typename T::FFTOutputType,
          typename Eigenvalues = typename T::EigenvalueOutputType,
          typename Eigenvectors = typename T::EigenvectorOutputType,
          typename BeamOutputType = typename T::BeamOutputType,
          typename EigenmodeCounts = std::nullptr_t>
class BufferedOutput : public Output {
public:
  BufferedOutput(
      std::unique_ptr<
          BeamWriter<BeamOutputType, typename T::ArrivalsOutputType,
                     EigenmodeCounts>>
          beam_writer,
      std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
          vis_writer,
      std::unique_ptr<
          EigenWriter<Eigenvalues, Eigenvectors, EigenmodeCounts>>
          eigen_writer,
      std::unique_ptr<FFTWriter<FFTOutput>> fft_writer)
      : beam_writer_(std::move(beam_writer)),
        vis_writer_(std::move(vis_writer)),
        eigen_writer_(std::move(eigen_writer)),
        fft_writer_(std::move(fft_writer)), running_(true) {};

  ~BufferedOutput() {
    running_ = false;

    if (beam_writer_ != nullptr) {
      beam_writer_->stop();
    };
    if (vis_writer_ != nullptr) {
      vis_writer_->stop();
    };
    if (eigen_writer_ != nullptr) {
      eigen_writer_->stop();
    };
    if (fft_writer_ != nullptr) {
      fft_writer_->stop();
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
                            const size_t end_seq_num) override {
    if (fft_writer_ != nullptr) {
      return fft_writer_->register_block(start_seq_num, end_seq_num);
    }
    return std::numeric_limits<size_t>::max();
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
    return vis_writer_->get_visibilities_landing_pointer(block_num);
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

  void *
  get_nulled_eigenmode_counts_landing_pointer(const size_t block_num) override {
    if (eigen_writer_ == nullptr) {
      return nullptr;
    }
    return eigen_writer_->get_nulled_eigenmode_counts_landing_pointer(
        block_num);
  }

  void *get_beam_counts_landing_pointer(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return nullptr;
    }
    return beam_writer_->get_nulled_eigenmode_counts_landing_pointer(block_num);
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
    beam_writer_->notify();
  }

  void
  register_visibilities_transfer_complete(const size_t block_num) override {
    if (vis_writer_ == nullptr) {
      return;
    }
    vis_writer_->register_visibilities_transfer_complete(block_num);
    vis_writer_->notify();
  }

  void register_arrivals_transfer_complete(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return;
    }
    beam_writer_->register_arrivals_transfer_complete(block_num);
    beam_writer_->notify();
  }

  void register_eigendecomposition_data_transfer_complete(
      const size_t block_num) override {
    if (eigen_writer_ == nullptr) {
      return;
    }
    eigen_writer_->register_eigendecomposition_transfer_complete(block_num);
    eigen_writer_->notify();
  }

  void
  register_beam_counts_transfer_complete(const size_t block_num) override {
    if (beam_writer_ == nullptr) {
      return;
    }
    beam_writer_->register_counts_transfer_complete(block_num);
    beam_writer_->notify();
  }

  void register_fft_transfer_complete(const size_t block_num) override {
    if (fft_writer_ == nullptr) {
      return;
    }
    fft_writer_->register_fft_transfer_complete(block_num);
    fft_writer_->notify();
  }

  void start_writer_loop() {
    INFO_LOG("Writer loop is running!");

    if (beam_writer_ != nullptr) {
      beam_writer_->start();
    }

    if (vis_writer_ != nullptr) {
      vis_writer_->start();
    }

    if (eigen_writer_ != nullptr) {
      eigen_writer_->start();
    }

    if (fft_writer_ != nullptr) {
      fft_writer_->start();
    }
  }

  void stop_writers() {

    if (beam_writer_ != nullptr) {
      beam_writer_->stop();
    }

    if (vis_writer_ != nullptr) {
      vis_writer_->stop();
    }

    if (eigen_writer_ != nullptr) {
      eigen_writer_->stop();
    }

    if (fft_writer_ != nullptr) {
      fft_writer_->stop();
    }
  }

  std::atomic<bool> running_{true};

private:
  std::unique_ptr<BeamWriter<BeamOutputType, typename T::ArrivalsOutputType,
                             EigenmodeCounts>>
      beam_writer_;
  std::unique_ptr<VisibilitiesWriter<typename T::VisibilitiesOutputType>>
      vis_writer_;
  std::unique_ptr<EigenWriter<Eigenvalues, Eigenvectors, EigenmodeCounts>>
      eigen_writer_;
  std::unique_ptr<FFTWriter<FFTOutput>> fft_writer_;
};
