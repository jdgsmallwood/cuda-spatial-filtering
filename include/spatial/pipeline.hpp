#pragma once
#include <iostream>
// Forward declaration of this.
class ProcessorStateBase;
struct FinalPacketData;

class GPUPipeline {
  // At some point in this class, we need to call release_buffer on
  // ProcessorState
public:
  void set_state(ProcessorStateBase *state) { state_ = state; };
  virtual void execute_pipeline(FinalPacketData *packet_data) = 0;

protected:
  ProcessorStateBase *state_;
};

class LambdaGPUPipeline : public GPUPipeline {
public:
  void execute_pipeline(FinalPacketData *packet_data) override;
};
