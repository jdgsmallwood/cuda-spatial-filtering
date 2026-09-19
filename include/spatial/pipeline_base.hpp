#pragma once
#include <stdexcept>
#include <vector>
// Forward declaration of this.
class ProcessorStateBase;
class Output;
struct FinalPacketData;

class GPUPipeline {
  // At some point in this class, we need to call release_buffer on
  // ProcessorState
public:
  void set_state(ProcessorStateBase *state) { state_ = state; };
  void set_output(std::shared_ptr<Output> output) { output_ = output; };
  virtual void set_subpacket_delays(int *delays_subpacket) {
    subpacket_delays_ = delays_subpacket;
  };
  virtual void set_antenna_gains(std::complex<float> *gains) {
    gains_ = gains;
  };
  virtual void set_fine_delays(const float * /*delays_ns*/, double /*base_freq_hz*/,
                               double /*channel_bw_hz*/, int /*min_freq_ch*/) {};
  virtual void set_stream_permutation(const std::vector<int> & /*recv_perm*/,
                                      const std::vector<int> & /*pol_perm*/) {
    throw std::runtime_error(
        "set_stream_permutation not supported by this pipeline type");
  }
  virtual void execute_pipeline(FinalPacketData *packet_data,
                                const bool dummy_run = false) = 0;
  virtual void dump_visibilities(const uint64_t end_seq_num = 0) = 0;

  // GPUDirect RDMA ingest support (see /home/ubuntu/.claude/plans/i-want-to-start-breezy-lampson.md
  // and docs/architecture.md's GPUDirect section): the device-resident
  // samples/scales array for one input buffer slot, so a capture backend
  // can relocate packet payloads directly into it instead of via a pinned-host
  // buffer + cudaMemcpyAsync. Default null means "no GPU-resident landing
  // buffer available" -- the CPU-memory ingest path (KernelSocketPacketCapture,
  // PCAPPacketCapture) never calls these and doesn't need an override.
  virtual void *gpu_landing_samples_ptr(int /*buffer_index*/) { return nullptr; }
  virtual void *gpu_landing_scales_ptr(int /*buffer_index*/) { return nullptr; }

protected:
  // Initialized so a pipeline used before set_state/set_output is called
  // (e.g. the constructor warmup run) trips ingest_and_scale's nullptr
  // guard instead of dereferencing an indeterminate pointer.
  ProcessorStateBase *state_ = nullptr;
  std::shared_ptr<Output> output_;
  int *subpacket_delays_ = nullptr;
  std::complex<float> *gains_ = nullptr;
};
