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

template <size_t NR_CHANNELS, size_t NR_POLARIZATIONS, size_t NR_BEAMS>
class SingleHostMemoryOutput : public Output {

private:
  using BeamOutput = __half[NR_CHANNELS][NR_POLARIZATIONS][NR_BEAMS];
  BeamOutput *data;

public:
  size_t register_block() override { return 1; };

  void *get_landing_pointer(size_t block_num) override { return (void *)data; };

  void register_transfer_complete(size_t block_num) override {};

  SingleHostMemoryOutput() {
    CUDA_CHECK(cudaMallocHost((void **)&data, sizeof(BeamOutput)));
  };
};
