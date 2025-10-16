#include "spatial/spatial.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <unistd.h>

class Output {
public:
  virtual size_t register_block() = 0;

  virtual void *get_landing_pointer(size_t block_num) = 0;

  virtual void register_transfer_complete(size_t block_num) = 0;
};

template <typename T> class SingleHostMemoryOutput : public Output {

public:
  using BeamOutput =
      float[T::NR_CHANNELS][T::NR_POLARIZATIONS][T::NR_BEAMS]
           [T::NR_PACKETS_FOR_CORRELATION * T::NR_TIME_STEPS_PER_PACKET][2];
  BeamOutput *data;

  size_t register_block() override { return 1; };

  void *get_landing_pointer(size_t block_num) override { return (void *)data; };

  void register_transfer_complete(size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&data, sizeof(BeamOutput)));
  };
};
