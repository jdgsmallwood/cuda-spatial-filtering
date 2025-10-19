#include "spatial/spatial.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unistd.h>

class Output {
public:
  virtual size_t register_beam_data_block() = 0;
  virtual size_t register_visibilities_block() = 0;

  virtual void *get_beam_data_landing_pointer(size_t block_num) = 0;
  virtual void *get_visibilities_landing_pointer(size_t block_num) = 0;

  virtual void register_beam_data_transfer_complete(size_t block_num) = 0;
  virtual void register_visibilities_transfer_complete(size_t block_num) = 0;
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
  BeamOutput *beam_data;
  Visibilities *visibilities;

  size_t register_beam_data_block() override { return 1; };
  size_t register_visibilities_block() override { return 1; };

  void *get_beam_data_landing_pointer(size_t block_num) override {
    return (void *)beam_data;
  };

  void *get_visibilities_landing_pointer(size_t block_num) override {
    return (void *)visibilities;
  }

  void register_beam_data_transfer_complete(size_t block_num) override {};
  void register_visibilities_transfer_complete(size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&beam_data, sizeof(BeamOutput)));
    CUDA_CHECK(cudaMallocHost((void **)&visibilities, sizeof(Visibilities)));
  };
  ~SingleHostMemoryOutput() {
    cudaFreeHost(beam_data);
    cudaFreeHost(visibilities);
  };
};
