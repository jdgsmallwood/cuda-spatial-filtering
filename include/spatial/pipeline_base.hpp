#pragma once
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
  virtual void execute_pipeline(FinalPacketData *packet_data,
                                const bool dummy_run = false) = 0;
  virtual void dump_visibilities(const uint64_t end_seq_num = 0) = 0;

protected:
  // Initialized so a pipeline used before set_state/set_output is called
  // (e.g. the constructor warmup run) trips ingest_and_scale's nullptr
  // guard instead of dereferencing an indeterminate pointer.
  ProcessorStateBase *state_ = nullptr;
  std::shared_ptr<Output> output_;
  int *subpacket_delays_ = nullptr;
  std::complex<float> *gains_ = nullptr;
};
