#include "spatial/ethernet.hpp"
#include "spatial/spatial.cuh"
#include "spatial/spatial.hpp"
#include "spatial/tensor.hpp"
#include <algorithm>
#include <atomic>
#include <ccglib/ccglib.hpp>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cudawrappers/cu.hpp>
#include <cutensor.h>
#include <iostream>
#include <libtcc/Correlator.h>
#include <memory>
#include <pcap/pcap.h>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#define DEBUG 1

constexpr int NUM_BUFFERS = 2;
constexpr int NR_ACTUAL_RECEIVERS = 20;
constexpr int NR_TIME_STEPS_PER_PACKET = 64;
constexpr int NR_ACTUAL_BASELINES =
    NR_ACTUAL_RECEIVERS * (NR_ACTUAL_RECEIVERS + 1) / 2;

PCAPInfo get_pcap_info(char *file_name) {

  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t *handle = pcap_open_offline(file_name, errbuf);
  if (!handle) {
    printf("pcap_open_offline failed: %s\n", errbuf);
  }

  struct pcap_pkthdr *header;
  const u_char *data;
  int res;

  // figure out what the data looks like.
  int start_freq_channel = -1;
  int end_freq_channel = -1;

  int start_seq_num = -1;
  int end_seq_num = -1;
  bool first_packet = true;

  int num_packets_captured = 0;
  while ((res = pcap_next_ex(handle, &header, &data)) >= 0) {
    if (res == 0)
      continue; // Timeout in live capture, ignore for offline
    num_packets_captured++;
    PacketInfo p = get_packet_info(data, header->len);

    if (first_packet) {
      start_seq_num = p.sample_count;
      end_seq_num = p.sample_count;

      start_freq_channel = p.freq_channel;
      end_freq_channel = p.freq_channel;
      first_packet = false;
    } else {
      if (p.sample_count < start_seq_num) {
        start_seq_num = p.sample_count;
      }

      if (p.sample_count > end_seq_num) {
        end_seq_num = p.sample_count;
      }

      if (p.freq_channel < start_freq_channel) {
        start_freq_channel = p.freq_channel;
      }

      if (p.freq_channel > end_freq_channel) {
        end_freq_channel = p.freq_channel;
      }
    }
  }

  PCAPInfo p = {start_freq_channel, end_freq_channel, start_seq_num,
                end_seq_num, num_packets_captured};
  if (res == -1) {
    printf("Error reading packets: %s\n", pcap_geterr(handle));
  }
  pcap_close(handle);
  return p;
}

int main(int argc, char *argv[]) {
  /* Read data from a PCAP file, run through the Tensor-Core Correlator and
   * output visibilities to stderr.
   *
   * Structure:
   * 1. Read through PCAP packets to get frequency channels, length of packets
   * etc.
   * 2. Read through again to store data.
   * 3. Load data to h_samples, h_visibilities
   * 4. Run CUDA operations & print
   *
   * We use a multiple buffer / stream system to allow stream concurrency on the
   * GPU.
   *
   * */

  // Check if any padding
  std::cout << "sizeof(__half): " << sizeof(__half) << std::endl;
  std::cout << "sizeof(std::complex<__half>): " << sizeof(std::complex<__half>)
            << std::endl;
  static_assert(sizeof(std::complex<__half>) == 2 * sizeof(__half),
                "Padding detected in std::complex<__half>");

  std::cout << "sizeof(float): " << sizeof(float) << std::endl;
  std::cout << "sizeof(std::complex<float>): " << sizeof(std::complex<float>)
            << std::endl;
  static_assert(sizeof(std::complex<float>) == 2 * sizeof(float),
                "Padding detected in std::complex<float>");

  /*
   * PCAP Formatting.
   * */

  if (argc < 2) {
    printf("Usage: %s <pcap file>\n", argv[0]);
    return 1;
  }

  PCAPInfo info = get_pcap_info(argv[1]);
  printf("Start freq channel is %u and end freq channel is %u\n",
         info.start_freq, info.end_freq);
  printf("Start seq num is %u and end seq num is %u\n", info.start_seq,
         info.end_seq);

  printf("Total packets captured is %u\n", info.num_packets_captured);
  const int num_freq = info.end_freq - info.start_freq + 1;
  const int number_of_aggregated_packets = std::max(
      info.num_packets_captured / NR_BLOCKS_FOR_CORRELATION / num_freq, 1);
  printf("Storing in groups of 10. Number of aggregated packets is %u\n",
         number_of_aggregated_packets);
  // second pass to actually get the data

  // Reopen the file to return to the beginning of the file.
  char errbuf[PCAP_ERRBUF_SIZE];
  pcap_t *handle = pcap_open_offline(argv[1], errbuf);

  struct pcap_pkthdr *header;
  const u_char *data;
  int res;
  /*
   * PCAP Data Reading
   */

  // allocate pinned host memory
  Samples *h_samples;
  Visibilities *h_visibilities;
  std::complex<__half> *h_weights;
  BeamformedData *h_beamformed_data;

  constexpr int num_weights =
      NUM_BEAMS * NR_RECEIVERS * NR_POLARIZATIONS * NR_CHANNELS;

  constexpr int num_eigen = NR_RECEIVERS * NR_CHANNELS * NR_POLARIZATIONS;

  cudaMallocHost(&h_samples, number_of_aggregated_packets * sizeof(Samples));
  cudaMallocHost(&h_visibilities,
                 number_of_aggregated_packets * sizeof(Visibilities));
  cudaMallocHost(&h_weights, num_weights * sizeof(std::complex<__half>));
  // not sure about this memory allocation.
  cudaMallocHost(&h_beamformed_data,
                 number_of_aggregated_packets * sizeof(BeamformedData));

  for (auto i = 0; i < num_weights; ++i) {
    h_weights[i] = std::complex<__half>(__float2half(1.0f), __float2half(1.0f));
  }

  std::vector<Tscale> scales(NR_ACTUAL_RECEIVERS);

  Tscale *h_scales;
  cudaMallocHost(&h_scales, NR_ACTUAL_BASELINES * sizeof(Tscale));

  printf("Setting h_samples & h_visibilities memory to zero\n");
  std::memset(h_samples, 0, number_of_aggregated_packets * sizeof(Samples));
  std::memset(h_visibilities, 0,
              number_of_aggregated_packets * sizeof(Visibilities));
  printf("Processing packets\n");
  while ((res = pcap_next_ex(handle, &header, &data)) >= 0) {
    if (res == 0)
      continue; // Timeout in live capture, ignore for offline
    process_packet(data, header->len, h_samples, scales, info.start_seq,
                   info.start_freq, NR_TIME_STEPS_PER_PACKET,
                   NR_BLOCKS_FOR_CORRELATION, NR_TIMES_PER_BLOCK,
                   NR_ACTUAL_RECEIVERS);
  }
  pcap_close(handle);

  printf("h_scales:\n");
  for (auto i = 0; i < NR_ACTUAL_RECEIVERS; ++i) {
    for (auto j = 0; j <= i; ++j) {
      int baseline = i * (i + 1) / 2 + j;
      h_scales[baseline] = scales[i] * scales[j];
      printf("baseline (%i, %i, %i): %u x %u = %u\n", i, j, baseline, scales[i],
             scales[j], h_scales[baseline]);
    }
  }

  // create CUDA streams
  cudaStream_t streams[NUM_BUFFERS];
  cudaEvent_t input_transfer_done[NUM_BUFFERS];

  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    cudaEventCreate(&input_transfer_done[i]);
  }

  // create device pointers
  Samples *d_samples[NUM_BUFFERS];

  // the planar data needs to be __half so if NR_BITS == 8 then
  // we need to convert
#if NR_BITS == 8
  __half *d_samples_converted[NUM_BUFFERS];
  float *d_visibilities_converted[NUM_BUFFERS];
#endif
  __half *d_samples_planar[NUM_BUFFERS];
  Visibilities *d_visibilities[NUM_BUFFERS];
  std::complex<float> *d_visibilities_permuted[NUM_BUFFERS];
  __half *d_weights[NUM_BUFFERS], *d_weights_updated[NUM_BUFFERS],
      *d_weights_permuted[NUM_BUFFERS];
  float *d_eigenvalues[NUM_BUFFERS];
  BeamformedData *d_beamformed_data[NUM_BUFFERS],
      *d_beamformed_data_output[NUM_BUFFERS];
  // start with these events in done state.
  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaMalloc((void **)&d_samples[i], sizeof(Samples));
    cudaMalloc((void **)&d_samples_planar[i], sizeof(Samples));
    cudaMalloc((void **)&d_visibilities[i], sizeof(Visibilities));
    cudaMalloc((void **)&d_visibilities_permuted[i], sizeof(Visibilities));
    cudaMalloc((void **)&d_weights[i],
               num_weights * sizeof(std::complex<__half>));
    cudaMalloc((void **)&d_weights_updated[i],
               num_weights * sizeof(std::complex<__half>));
    cudaMalloc((void **)&d_weights_permuted[i],
               num_weights * sizeof(std::complex<__half>));
    cudaMalloc((void **)&d_eigenvalues[i], sizeof(float) * num_eigen);
    cudaMalloc((void **)&d_beamformed_data[i], sizeof(BeamformedData));
    cudaMalloc((void **)&d_beamformed_data_output[i], sizeof(BeamformedData));

#if NR_BITS == 8
    cudaMalloc((void **)&d_samples_converted[i],
               sizeof(Samples) / sizeof(int8_t) * sizeof(__half));
    cudaMalloc((void **)&d_visibilities_converted[i],
               sizeof(Visibilities) / sizeof(int) * sizeof(float));
#endif

    // transfer weights
    cudaMemcpy(d_weights[i], h_weights,
               sizeof(std::complex<__half>) * num_weights, cudaMemcpyDefault);
    cudaEventRecord(input_transfer_done[i], streams[i]);
  }

#if DEBUG == 1
  // allocate debug buffers
  __half *h_weights_updated, *h_weights_permuted, *h_samples_planar,
      *h_weights_check;

  cudaMallocHost(&h_weights_updated,
                 num_weights * sizeof(std::complex<__half>));
  cudaMallocHost(&h_weights_permuted,
                 num_weights * sizeof(std::complex<__half>));
  cudaMallocHost(&h_samples_planar, sizeof(Samples));
  cudaMallocHost(&h_weights_check, num_weights * sizeof(std::complex<__half>));
#endif

  const __half alpha = __float2half(1.0f);
  const float alpha_32 = 1.0f;
  // c = channel
  // b = block
  // r = receiver
  // p = polarization
  // t = time
  // z = complex
  // l = baseline
  // m = beam
  std::vector<int> modePacket{'c', 'b', 'r', 'p', 't', 'z'};
  std::vector<int> modePlanar{'c', 'p', 'z', 'r', 'b', 't'};
  std::vector<int> modeVisCorr{'c', 'l', 'p', 'q', 'z'};
  std::vector<int> modeVisDecomp{'c', 'p', 'q', 'l', 'z'};
  std::vector<int> modeBeamCCGLIB{'c', 'p', 'z', 'm', 'b', 't'};
  std::vector<int> modeBeamOutput{'c', 'p', 'm', 'b', 't', 'z'};
  std::vector<int> modeWeightsInput{'c', 'p', 'm', 'r', 'z'};
  std::vector<int> modeWeightsCCGLIB{'c', 'p', 'z', 'm', 'r'};

  std::unordered_map<int, int64_t> extent;
  extent['c'] = NR_CHANNELS;
  extent['b'] = NR_BLOCKS_FOR_CORRELATION;
  extent['r'] = NR_RECEIVERS;
  extent['p'] = NR_POLARIZATIONS;
  extent['q'] = NR_POLARIZATIONS; // 2nd polarizations for baselines
  extent['t'] = NR_TIMES_PER_BLOCK;
  extent['z'] = 2; // real, imaginary
  extent['l'] = NR_BASELINES;
  extent['m'] = NUM_BEAMS;

  CutensorSetup tensor_16(extent, CUTENSOR_R_16F, 128);
  CutensorSetup tensor_32(extent, CUTENSOR_R_32F, 128);

  tensor_16.addTensor(modePacket, "packet");
  tensor_16.addTensor(modePlanar, "planar");

  tensor_16.addTensor(modeWeightsInput, "weightsInput");
  tensor_16.addTensor(modeWeightsCCGLIB, "weightsCCGLIB");
  tensor_32.addTensor(modeVisCorr, "visCorr");
  tensor_32.addTensor(modeVisDecomp, "visDecomp");

  tensor_32.addTensor(modeBeamCCGLIB, "beamCCGLIB");
  tensor_32.addTensor(modeBeamOutput, "beamOutput");

  tensor_16.addPermutation("packet", "planar", CUTENSOR_COMPUTE_DESC_16F,
                           "packetToPlanar");
  tensor_16.addPermutation("weightsInput", "weightsCCGLIB",
                           CUTENSOR_COMPUTE_DESC_16F, "weightsInputToCCGLIB");
  tensor_32.addPermutation("visCorr", "visDecomp", CUTENSOR_COMPUTE_DESC_32F,
                           "visCorrToDecomp");
  tensor_32.addPermutation("beamCCGLIB", "beamOutput",
                           CUTENSOR_COMPUTE_DESC_32F, "beamCCGLIBToOutput");
  printf("Initializing correlator\n");
  tcc::Correlator correlator(cu::Device(0), inputFormat, NR_RECEIVERS,
                             NR_CHANNELS, NR_SAMPLES_PER_CHANNEL,
                             NR_POLARIZATIONS, NR_RECEIVERS_PER_BLOCK);

  printf("NR_RECEIVERS: %u\n", NR_RECEIVERS);
  printf("NR_CHANNELS: %u\n", NR_CHANNELS);
  printf("NR_SAMPLES_PER_CHANNEL: %u\n", NR_SAMPLES_PER_CHANNEL);
  printf("NR_TIMES_PER_BLOCK: %u\n", NR_TIMES_PER_BLOCK);
  printf("NR_POLARIZATIONS: %u\n", NR_POLARIZATIONS);
  printf("NR_RECEIVERS_PER_BLOCK: %u\n", NR_RECEIVERS_PER_BLOCK);
  printf("NR_ACTUAL_RECEIVERS: %u\n", NR_ACTUAL_RECEIVERS);
  printf("NR_BITS: %u\n", NR_BITS);
  printf("Launching processing loop...\n");
  int current_buffer = 0;
  // std::atomic is overkill right now but if we end up using multi-threading at
  // some point this sidesteps a race condition.
  std::atomic<int> last_frame_processed = 0;
  bool processing = true;

  CUdevice cu_device;
  cuDeviceGet(&cu_device, 0);

  std::vector<std::unique_ptr<ccglib::mma::GEMM>> gemm_handles;

  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    gemm_handles.emplace_back(std::make_unique<ccglib::mma::GEMM>(
        NR_CHANNELS * NR_POLARIZATIONS, NUM_BEAMS,
        NR_TIMES_PER_BLOCK * NR_BLOCKS_FOR_CORRELATION, NR_RECEIVERS, cu_device,
        streams[i], ccglib::ValueType::float16, ccglib::mma::basic));
  }
  // Ensure all copying is done before processing loop starts.
  // This may not be necessary.
  cudaDeviceSynchronize();
  // Main processing loop.
  while (processing) {
    if (cudaEventQuery(input_transfer_done[current_buffer]) == cudaSuccess) {
      printf("Beginning new processing loop....\n");
      int next_frame_to_capture = last_frame_processed.fetch_add(1);
      printf("Next frame to capture is %u for stream %u\n",
             next_frame_to_capture, current_buffer);
      // Use this for the lambda function to capture current value.
      if (next_frame_to_capture + 1 >= number_of_aggregated_packets) {
        processing = false;
        printf(
            "Finishing processing loop as next frame is %u which +1 is greater "
            "than or equal to the number of aggregated_packets %u\n",
            next_frame_to_capture, number_of_aggregated_packets);
      }

      cudaMemcpyAsync(d_samples[current_buffer],
                      h_samples[next_frame_to_capture], sizeof(Samples),
                      cudaMemcpyHostToDevice, streams[current_buffer]);
      // Now we can start preparing the next buffer for transport to the GPU.
      cudaEventRecord(input_transfer_done[current_buffer],
                      streams[current_buffer]);

      correlator.launchAsync((CUstream)streams[current_buffer],
                             (CUdeviceptr)d_visibilities[current_buffer],
                             (CUdeviceptr)d_samples[current_buffer]);

#if NR_BITS == 8
      convert_int8_to_half((int8_t *)d_samples[current_buffer],
                           d_samples_converted[current_buffer],
                           sizeof(Samples) / sizeof(int8_t),
                           streams[current_buffer]);
      convert_int_to_float((int *)d_visibilities[current_buffer],
                           d_visibilities_converted[current_buffer],
                           sizeof(Visibilities) / sizeof(int),
                           streams[current_buffer]);

      tensor_16.runPermutation(
          "packetToPlanar", alpha, d_samples_converted[current_buffer],
          d_samples_planar[current_buffer], streams[current_buffer]);

      tensor_32.runPermutation(
          "visCorrToDecomp", alpha_32,
          (float *)d_visibilities_converted[current_buffer],
          (float *)d_visibilities_permuted[current_buffer],
          streams[current_buffer]);
#elif NR_BITS == 16
      tensor_16.runPermutation(
          "packetToPlanar", alpha, d_samples[current_buffer],
          d_samples_planar[current_buffer], streams[current_buffer]);
      tensor_32.runPermutation("visCorrToDecomp", alpha_32,
                               (float *)d_visibilities[current_buffer],
                               (float *)d_visibilities_permuted[current_buffer],
                               streams[current_buffer]);
#endif

      checkCudaCall(cudaMemcpyAsync(
          h_visibilities[next_frame_to_capture], d_visibilities[current_buffer],
          sizeof(Visibilities), cudaMemcpyDefault, streams[current_buffer]));
      // need to think how multiple channels / polarizations works here - do we
      // need to do multiple decompositions? Probably yes. We'll also need to
      // convert the visibilities from int32 -> float

      // d_eigendecomposition(d_eigenvalues[current_buffer], NR_RECEIVERS,
      //                      NR_CHANNELS, NR_POLARIZATIONS,
      //                     d_visibilities_permuted[current_buffer],
      //                   streams[current_buffer]);

#if NR_BITS == 8
      update_weights(
          d_weights[current_buffer], d_weights_updated[current_buffer],
          NUM_BEAMS, NR_RECEIVERS, NR_CHANNELS, NR_POLARIZATIONS,
          d_eigenvalues[current_buffer],
          d_visibilities_converted[current_buffer], streams[current_buffer]);

#else

      update_weights(d_weights[current_buffer],
                     d_weights_updated[current_buffer], NUM_BEAMS, NR_RECEIVERS,
                     NR_CHANNELS, NR_POLARIZATIONS,
                     d_eigenvalues[current_buffer],
                     d_visibilities[current_buffer], streams[current_buffer]);
#endif

      tensor_16.runPermutation("weightsInputToCCGLIB", alpha,
                               (__half *)d_weights_updated[current_buffer],
                               d_weights_permuted[current_buffer],
                               streams[current_buffer]);
#if DEBUG == 1
      cudaMemcpyAsync(h_weights_updated, d_weights_updated[current_buffer],
                      sizeof(std::complex<__half>) * num_weights,
                      cudaMemcpyDefault, streams[current_buffer]);
      cudaMemcpyAsync(h_weights_permuted, d_weights_permuted[current_buffer],
                      sizeof(std::complex<__half>) * num_weights,
                      cudaMemcpyDefault, streams[current_buffer]);
      cudaMemcpyAsync(h_samples_planar, d_samples_planar[current_buffer],
                      sizeof(Samples), cudaMemcpyDefault,
                      streams[current_buffer]);
      cudaMemcpyAsync(h_weights_check, d_weights[current_buffer],
                      sizeof(std::complex<__half>) * num_weights,
                      cudaMemcpyDefault, streams[current_buffer]);
#endif

      (*gemm_handles[current_buffer])
          .Run((CUdeviceptr)d_weights_permuted[current_buffer],
               (CUdeviceptr)d_samples_planar[current_buffer],
               (CUdeviceptr)d_beamformed_data[current_buffer]);

      tensor_32.runPermutation(
          "beamCCGLIBToOutput", alpha_32,
          (float *)d_beamformed_data[current_buffer],
          (float *)d_beamformed_data_output[current_buffer],
          streams[current_buffer]);

      cudaMemcpyAsync(&h_beamformed_data[next_frame_to_capture],
                      d_beamformed_data_output[current_buffer],
                      sizeof(BeamformedData), cudaMemcpyDefault,
                      streams[current_buffer]);
    }

    current_buffer = (current_buffer + 1) % NUM_BUFFERS;
  }
  printf("Synchronizing...\n");
  cudaDeviceSynchronize();

  /*
   * Output
   * */

  printf("Starting to print visibilities...\n");
  for (auto i = 0; i < number_of_aggregated_packets; ++i) {
    printf("Visibilities for %u:\n", i);
    print_nonzero_visibilities(&h_visibilities[i], h_scales);

    printf("Beams for %u:\n", i);
    print_nonzero_beams(&h_beamformed_data[i], NR_CHANNELS, NR_POLARIZATIONS,
                        NUM_BEAMS,
                        NR_BLOCKS_FOR_CORRELATION * NR_TIMES_PER_BLOCK);
  }

#if DEBUG == 1
  printf("weights original...\n");
  for (auto i = 0; i < num_weights; ++i) {
    const std::complex<__half> val = h_weights[i];
    printf("%u: %f + %f j\n", i, __half2float(val.real()),
           __half2float(val.imag()));
  }

  printf("weights check...\n");
  for (auto i = 0; i < num_weights; ++i) {
    const std::complex<__half> val = std::complex<__half>(
        h_weights_check[2 * i], h_weights_check[2 * i + 1]);
    printf("%u: %f + %f j\n", i, __half2float(val.real()),
           __half2float(val.imag()));
  }
  printf("weights updated...\n");
  for (auto i = 0; i < num_weights; ++i) {
    const std::complex<__half> val = std::complex<__half>{
        h_weights_updated[2 * i], h_weights_updated[2 * i + 1]};
    printf("%u: %f + %f j\n", i, __half2float(val.real()),
           __half2float(val.imag()));
  }

  printf("weights permuted...\n");

  for (auto i = 0; i < num_weights * 2; ++i) {
    const __half weight = h_weights_permuted[i];
    printf("%u: %f\n", i, __half2float(weight));
  }
#endif

  for (auto i = 0; i < NUM_BUFFERS; ++i) {
    cudaFree(d_samples[i]);
    cudaFree(d_samples_planar[i]);
    cudaFree(d_visibilities[i]);
    cudaFree(d_beamformed_data[i]);
    cudaFree(d_beamformed_data_output[i]);
    cudaFree(d_eigenvalues[i]);
    cudaFree(d_weights[i]);
    cudaFree(d_weights_updated[i]);
    cudaFree(d_weights_permuted[i]);
    cudaStreamDestroy(streams[i]);
    cudaEventDestroy(input_transfer_done[i]);
  }
  cudaFreeHost(h_samples);
  cudaFreeHost(h_visibilities);
  cudaFreeHost(h_scales);
  cudaFreeHost(h_beamformed_data);

  return 0;
}
