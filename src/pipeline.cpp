#include "spatial/pipeline.hpp"
#include "spatial/logging.hpp"
#include "spatial/packet_formats.hpp"
#include "spatial/spatial.hpp"
#include <iostream>

void LambdaGPUPipeline::execute_pipeline(FinalPacketData *packet_data) {
  LOG_INFO("Hello from LambdaGPUPipeline!");
  if (state_ == nullptr) {
    std::logic_error("State has not been set on GPUPipeline object!");
  }
  LOG_INFO("Releasing buffer #{}", packet_data->buffer_index);
  state_->release_buffer(packet_data->buffer_index);
};
