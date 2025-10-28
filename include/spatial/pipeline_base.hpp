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
  virtual void execute_pipeline(FinalPacketData *packet_data) = 0;
  virtual void dump_visibilities(const int end_seq_num = -1) = 0;

protected:
  ProcessorStateBase *state_;
  std::shared_ptr<Output> output_;
};
