#pragma once

#include "spatial/logging.hpp"
#include "spatial/pinned_vector.hpp"
#include <array>
#include <atomic>
#include <charconv>
#include <condition_variable>
#include <fitsio.h>
#include <fstream>
#include <hdf5.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <immintrin.h>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <casacore/casa/Arrays/ArrayMath.h>
#include <casacore/casa/Arrays/Matrix.h>
#include <casacore/casa/Arrays/Vector.h>
#include <casacore/measures/Measures/MDirection.h>
#include <casacore/ms/MeasurementSets/MSColumns.h>
#include <casacore/ms/MeasurementSets/MeasurementSet.h>
#include <casacore/tables/Tables/ArrColDesc.h>
#include <casacore/tables/Tables/ScaColDesc.h>
#include <casacore/tables/Tables/SetupNewTab.h>
#include <casacore/tables/Tables/TableDesc.h>
#include <sw/redis++/redis++.h>

#include <getopt.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>

extern "C" {
#include "ascii_header.h"
#include "dada_def.h"
#include "dada_hdu.h"
#include "futils.h"
#include "ipcio.h"
#include "multilog.h"
}

template <typename T> struct hdf5_storage_type {
  using type = T;
};

template <> struct hdf5_storage_type<__half> {
  using type = uint16_t;
};

template <typename BeamOutputType, typename ArrivalsOutputType>
struct BeamBlock {
  BeamOutputType beam_data;
  ArrivalsOutputType arrivals_data;
  size_t start_seq_num = 0;
  size_t end_seq_num = 0;

  bool beam_transfer_complete = false;
  bool arrival_transfer_complete = false;

  // Standardized interface for the base Writer class
  bool is_ready() const {
    return beam_transfer_complete && arrival_transfer_complete;
  }
  void reset_state() {
    beam_transfer_complete = false;
    arrival_transfer_complete = false;
  }
};

template <typename VisibilitiesOutputType> struct VisBlock {
  VisibilitiesOutputType data;
  size_t start_seq_num;
  size_t end_seq_num;
  int num_missing_packets;
  int num_total_packets;
  bool transfer_complete;

  VisBlock()
      : start_seq_num(0), end_seq_num(0), num_missing_packets(0),
        num_total_packets(0), transfer_complete(false) {}

  bool is_ready() const { return transfer_complete; }

  void reset_state() { transfer_complete = false; }
};

template <typename Eigenvalues, typename Eigenvectors> struct EigenBlock {
  Eigenvalues eigenvalues;
  Eigenvectors eigenvectors;
  size_t start_seq_num;
  size_t end_seq_num;
  bool transfer_complete;

  EigenBlock() : start_seq_num(0), end_seq_num(0), transfer_complete(false) {};

  bool is_ready() const { return transfer_complete; }
  void reset_state() { transfer_complete = false; }
};

template <typename FFTOutputType> struct FFTBlock {

  FFTOutputType fft_output;
  size_t start_seq_num;
  size_t end_seq_num;
  bool transfer_complete;

  FFTBlock() : start_seq_num(0), end_seq_num(0), transfer_complete(false) {};

  void *data_landing_pointer() { return (void *)&fft_output; };

  bool is_ready() const { return transfer_complete; };
  void reset_state() { transfer_complete = false; };
};

template <typename T> class Writer {
public:
  Writer(const size_t buffer_size)
      : buffer_size_(buffer_size), read_idx_(0), write_idx_(0) {
    blocks_.resize(buffer_size);
  }

  virtual ~Writer() { stop(); }

  void start() {
    running_.store(true, std::memory_order_release);
    drain_thread_ = std::thread(&Writer::drain_loop, this);
  }

  void stop() {
    if (running_.exchange(false, std::memory_order_acq_rel)) {
      cv_.notify_all();
      if (drain_thread_.joinable()) {
        drain_thread_.join();
      }
      flush();
    }
  }

  void notify() { cv_.notify_one(); }

  virtual void process_block(const T &block) = 0;
  virtual void flush() = 0;
  virtual const char *writer_name() const = 0;

  virtual size_t register_block() {

    size_t block_num = write_idx_.load(std::memory_order_relaxed);
    blocks_[block_num].reset_state();
    if ((write_idx_ + 1) % buffer_size_ ==
        read_idx_.load(std::memory_order_acquire)) {
      handle_buffer_full();
    }
    write_idx_.store((block_num + 1) % buffer_size_, std::memory_order_relaxed);
    return block_num;
  }

  T &get_block(size_t index) { return blocks_[index]; }

  void drain_ready_blocks() {
    size_t current_write_idx_ = write_idx_.load(std::memory_order_acquire);
    size_t read = read_idx_.load(std::memory_order_relaxed);

    while (read != current_write_idx_ && blocks_[read].is_ready()) {
      process_block(blocks_[read]);
      read = (read + 1) % buffer_size_;
    }
    read_idx_.store(read);
  }

  bool has_data_to_write() const {
    return read_idx_ != write_idx_ && blocks_[read_idx_].is_ready();
  }

protected:
  virtual void handle_buffer_full() {
    ERROR_LOG("{} ring buffer is full. Waiting...", std::string(writer_name()));
    while ((write_idx_.load(std::memory_order_acquire) + 1) % buffer_size_ ==
           read_idx_.load(std::memory_order_acquire)) {
      _mm_pause();
    }
  }

  size_t buffer_size_;
  std::atomic<size_t> read_idx_;
  std::atomic<size_t> write_idx_;
  cuda_util::PinnedVector<T> blocks_;

private:
  void drain_loop() {
    while (true) {
      {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] {
          return !running_.load(std::memory_order_acquire) ||
                 has_data_to_write();
        });
      }
      while (has_data_to_write()) {
        drain_ready_blocks();
      }

      if (!running_.load(std::memory_order_acquire) && !has_data_to_write()) {
        break;
      }
    }
  }

  // Must be value-initialized: ~Writer() -> stop() reads running_ to decide
  // whether to join the drain thread and do a final flush(). Left
  // uninitialized, a writer that was never start()ed would nondeterministically
  // run that teardown (spurious flush on a writer with no live drain thread),
  // which is UB and was observed throwing from a destructor-time flush().
  std::atomic<bool> running_{false};
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread drain_thread_;
};

template <typename BeamT, typename ArrivalsT>
class BeamWriter : public Writer<BeamBlock<BeamT, ArrivalsT>> {
public:
  using BlockType = BeamBlock<BeamT, ArrivalsT>;
  BeamWriter(const int num_blocks = 100) : Writer<BlockType>(num_blocks) {};
  virtual ~BeamWriter() = default;
  virtual const char *writer_name() const override { return "BeamWriter"; };
  using Writer<BlockType>::register_block;
  virtual size_t register_block(const size_t start_seq_num,
                                const size_t end_seq_num) {
    size_t block_idx = Writer<BlockType>::register_block();
    this->blocks_[block_idx].start_seq_num = start_seq_num;
    this->blocks_[block_idx].end_seq_num = end_seq_num;
    return block_idx;
  }

  void *get_beam_data_landing_pointer(const size_t block_idx) {
    return (void *)&this->blocks_[block_idx].beam_data;
  }
  void *get_arrivals_data_landing_pointer(const size_t block_idx) {
    return (void *)&this->blocks_[block_idx].arrivals_data;
  }

  void register_beam_data_transfer_complete(const size_t block_num) {
    this->blocks_[block_num].beam_transfer_complete = true;
  }

  void register_arrivals_transfer_complete(const size_t block_num) {
    this->blocks_[block_num].arrival_transfer_complete = true;
  }

  virtual void flush() = 0;
};

template <typename VisibilitiesOutputType>
class VisibilitiesWriter : public Writer<VisBlock<VisibilitiesOutputType>> {
public:
  using BlockType = VisBlock<VisibilitiesOutputType>;
  VisibilitiesWriter(const int num_blocks = 100)
      : Writer<BlockType>(num_blocks) {};
  virtual ~VisibilitiesWriter() = default;
  virtual const char *writer_name() const override { return "VisWriter"; };
  using Writer<BlockType>::register_block;
  virtual size_t register_block(const size_t start_seq_num,
                                const size_t end_seq_num,
                                const int num_missing_packets,
                                const int num_total_packets) {
    size_t block_idx = Writer<BlockType>::register_block();
    this->blocks_[block_idx].start_seq_num = start_seq_num;
    this->blocks_[block_idx].end_seq_num = end_seq_num;
    this->blocks_[block_idx].num_missing_packets = num_missing_packets;
    this->blocks_[block_idx].num_total_packets = num_total_packets;
    return block_idx;
  }
  virtual void *get_visibilities_landing_pointer(const size_t block_num) {
    return (void *)&this->blocks_[block_num].data;
  }
  void register_visibilities_transfer_complete(const size_t block_num) {
    this->blocks_[block_num].transfer_complete = true;
  }

  virtual void flush() = 0;
};

template <typename TVal, typename TVec>
class EigenWriter : public Writer<EigenBlock<TVal, TVec>> {
public:
  using BlockType = EigenBlock<TVal, TVec>;
  EigenWriter(const int num_blocks = 100) : Writer<BlockType>(num_blocks) {};
  virtual ~EigenWriter() = default;
  virtual const char *writer_name() const override { return "EigenWriter"; };
  using Writer<BlockType>::register_block;
  virtual size_t register_block(const size_t start_seq_num,
                                const size_t end_seq_num) {
    size_t block_idx = Writer<BlockType>::register_block();
    this->blocks_[block_idx].start_seq_num = start_seq_num;
    this->blocks_[block_idx].end_seq_num = end_seq_num;
    return block_idx;
  }

  virtual void *get_eigenvectors_landing_pointer(const size_t block_num) {
    return (void *)&this->blocks_[block_num].eigenvectors;
  }

  virtual void *get_eigenvalues_landing_pointer(const size_t block_num) {
    return (void *)&this->blocks_[block_num].eigenvalues;
  }

  void register_eigendecomposition_transfer_complete(const size_t block_num) {
    this->blocks_[block_num].transfer_complete = true;
  }

  virtual void flush() = 0;
};

template <typename T> class FFTWriter : public Writer<FFTBlock<T>> {
public:
  using BlockType = FFTBlock<T>;
  FFTWriter(const int num_blocks = 100) : Writer<BlockType>(num_blocks) {};
  virtual ~FFTWriter() = default;
  virtual const char *writer_name() const override { return "FFTWriter"; };
  using Writer<BlockType>::register_block;
  virtual size_t register_block(const size_t start_seq_num,
                                const size_t end_seq_num) {
    size_t block_idx = Writer<BlockType>::register_block();
    this->blocks_[block_idx].start_seq_num = start_seq_num;
    this->blocks_[block_idx].end_seq_num = end_seq_num;
    return block_idx;
  }
  void register_fft_transfer_complete(const size_t block_num) {
    this->blocks_[block_num].transfer_complete = true;
  }

  virtual void *get_fft_landing_pointer(const size_t block_num) {
    return (void *)&this->blocks_[block_num].fft_output;
  }

  virtual void flush() = 0;
};

template <typename T, typename U = size_t, size_t N = 0>
constexpr auto get_array_dims() {
  if constexpr (std::is_array_v<T>) {
    constexpr U extent = std::extent_v<T>;
    auto inner = get_array_dims<std::remove_extent_t<T>, U, N + 1>();
    inner.insert(inner.begin(), extent);
    return inner;
  } else {
    return std::vector<U>{};
  }
}

template <typename BeamT, typename ArrivalsT>
class HDF5BeamWriter : public BeamWriter<BeamT, ArrivalsT> {

  using beam_type = typename std::remove_all_extents<BeamT>::type;
  using beam_storage_type = typename hdf5_storage_type<beam_type>::type;
  using arrival_type = typename std::remove_all_extents<ArrivalsT>::type;

public:
  HDF5BeamWriter(HighFive::File &file, const int num_blocks = 100)
      : BeamWriter<BeamT, ArrivalsT>(num_blocks), file_(file) {
    using namespace HighFive;
    beam_element_count_ = sizeof(BeamT) / sizeof(beam_type);
    arrivals_element_count_ = sizeof(ArrivalsT) / sizeof(bool);
    beam_dims_ = get_array_dims<BeamT>();
    arrivals_dims_ = get_array_dims<ArrivalsT>();

    std::vector<size_t> beam_dataset_dims = {0};
    std::vector<size_t> beam_dataset_max_dims = {DataSpace::UNLIMITED};

    beam_dataset_dims.insert(beam_dataset_dims.end(), beam_dims_.begin(),
                             beam_dims_.end());
    beam_dataset_max_dims.insert(beam_dataset_max_dims.end(),
                                 beam_dims_.begin(), beam_dims_.end());
    std::vector<hsize_t> beam_chunk = {1};
    beam_chunk.insert(beam_chunk.end(), beam_dims_.begin(), beam_dims_.end());

    std::vector<size_t> arrivals_dataset_dims = {0};
    std::vector<size_t> arrivals_dataset_max_dims = {DataSpace::UNLIMITED};

    arrivals_dataset_dims.insert(arrivals_dataset_dims.end(),
                                 arrivals_dims_.begin(), arrivals_dims_.end());
    arrivals_dataset_max_dims.insert(arrivals_dataset_max_dims.end(),
                                     arrivals_dims_.begin(),
                                     arrivals_dims_.end());
    std::vector<hsize_t> arrivals_chunk = {1};
    arrivals_chunk.insert(arrivals_chunk.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());
    // Create beam dataset
    DataSpace beam_space(beam_dataset_dims, beam_dataset_max_dims);
    DataSetCreateProps beam_props;
    beam_props.add(Chunking(beam_chunk));
    beam_props.add(Shuffle());
    beam_props.add(Deflate(1));
    beam_dataset_ = file_.createDataSet<beam_storage_type>(
        "beam_data", beam_space, beam_props);

    // Create arrivals dataset
    DataSpace arrivals_space(arrivals_dataset_dims, arrivals_dataset_max_dims);
    DataSetCreateProps arrivals_props;
    arrivals_props.add(Chunking(arrivals_chunk));
    arrivals_dataset_ = file_.createDataSet<arrival_type>(
        "arrivals", arrivals_space, arrivals_props);

    DataSetCreateProps beam_seq_props;
    beam_seq_props.add(Chunking(std::vector<hsize_t>{1, 2}));
    // Create sequence number dataset
    beam_seq_dataset_ = file_.createDataSet<size_t>(
        "beam_seq_nums", DataSpace({0, 2}, {DataSpace::UNLIMITED, 2}),
        beam_seq_props);
  }

  void process_block(
      const typename BeamWriter<BeamT, ArrivalsT>::BlockType &block) override {
    INFO_LOG("Writing beam block...");
    const size_t n = n_blocks_written_;

    std::vector<size_t> new_dims = {n + 1};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    beam_dataset_.resize(new_dims);

    std::vector<size_t> beam_offset = {n};
    beam_offset.insert(beam_offset.end(), beam_dims_.size(), 0);
    std::vector<size_t> beam_count = {1};
    beam_count.insert(beam_count.end(), beam_dims_.begin(), beam_dims_.end());

    if constexpr (std::is_same_v<beam_type, __half>) {
      beam_dataset_.select(beam_offset, beam_count)
          .write_raw(reinterpret_cast<const uint16_t *>(&block.beam_data[0]));
    } else {
      beam_dataset_.select(beam_offset, beam_count)
          .write_raw(&block.beam_data[0]);
    }

    std::vector<size_t> arrivals_new_dims = {n + 1};
    arrivals_new_dims.insert(arrivals_new_dims.end(), arrivals_dims_.begin(),
                             arrivals_dims_.end());
    std::vector<size_t> arrivals_offset = {n};
    arrivals_offset.insert(arrivals_offset.end(), arrivals_dims_.size(), 0);
    std::vector<size_t> arrivals_count = {1};
    arrivals_count.insert(arrivals_count.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());
    arrivals_dataset_.resize(arrivals_new_dims);
    arrivals_dataset_.select(arrivals_offset, arrivals_count)
        .write_raw(&block.arrivals_data[0]);

    beam_seq_dataset_.resize({n + 1, 2});
    std::vector<size_t> seq_nums = {block.start_seq_num, block.end_seq_num};
    beam_seq_dataset_.select({n, 0}, {1, 2}).write_raw(seq_nums.data());

    ++n_blocks_written_;
  }

  void flush() override { file_.flush(); }

private:
  HighFive::File &file_;
  size_t beam_element_count_;
  std::vector<size_t> beam_dims_;
  size_t arrivals_element_count_;
  std::vector<size_t> arrivals_dims_;
  HighFive::DataSet beam_dataset_;
  HighFive::DataSet arrivals_dataset_;
  HighFive::DataSet beam_seq_dataset_;
  size_t n_blocks_written_ = 0;
};

template <typename T> class HDF5FFTWriter : public FFTWriter<T> {
public:
  HDF5FFTWriter(HighFive::File &file, const int min_channel,
                const int max_channel,
                const std::unordered_map<int, int> *antenna_map = nullptr,
                const int num_blocks = 100)
      : FFTWriter<T>(num_blocks), file_(file),
        element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)) {
    using namespace HighFive;

    double start_time = std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch())
                                .count() /
                            86400.0 +
                        40587.0;
    file_.createAttribute<double>("mjd_start", start_time);
    file_.createAttribute<int>("min_channel", min_channel);
    file_.createAttribute<int>("max_channel", max_channel);

    fft_dims_ = get_array_dims<T>();
    std::vector<size_t> fft_dataset_dims = {0};
    std::vector<size_t> fft_dataset_max_dims = {DataSpace::UNLIMITED};

    fft_dataset_dims.insert(fft_dataset_dims.end(), fft_dims_.begin(),
                            fft_dims_.end());
    fft_dataset_max_dims.insert(fft_dataset_max_dims.end(), fft_dims_.begin(),
                                fft_dims_.end());
    std::vector<hsize_t> fft_chunk = {1};
    fft_chunk.insert(fft_chunk.end(), fft_dims_.begin(), fft_dims_.end());
    DataSpace fft_space(fft_dataset_dims, fft_dataset_max_dims);
    DataSetCreateProps props;
    props.add(HighFive::Chunking(fft_chunk));
    // I imagine createDataSet needs a primitive type
    fft_dataset_ = file_.createDataSet<float>("ffts", fft_space, props);

    DataSetCreateProps fft_seq_props;
    fft_seq_props.add(Chunking(std::vector<hsize_t>{1, 2}));
    fft_seq_dataset_ = file_.createDataSet<size_t>(
        "fft_seq_nums", DataSpace({0, 2}, {HighFive::DataSpace::UNLIMITED, 2}),
        fft_seq_props);

    // DataSetCreateProps fft_missing_props;
    // fft_missing_props.add(Chunking(std::vector<hsize_t>{1, 3}));
    // fft_missing_dataset_ = file_.createDataSet<float>(
    //     "fft_missing_nums",
    //     DataSpace({0, 3}, {HighFive::DataSpace::UNLIMITED, 3}),
    //     fft_missing_props);
  }

  void process_block(const FFTBlock<T> &block) override {
    const size_t n = n_blocks_written_;
    std::vector<size_t> new_dims = {n + 1};
    new_dims.insert(new_dims.end(), fft_dims_.begin(), fft_dims_.end());
    fft_dataset_.resize(new_dims);

    std::vector<size_t> fft_offset = {n};
    fft_offset.insert(fft_offset.end(), fft_dims_.size(), 0);
    std::vector<size_t> fft_count = {1};
    fft_count.insert(fft_count.end(), fft_dims_.begin(), fft_dims_.end());

    fft_dataset_.select(fft_offset, fft_count)
        .write(reinterpret_cast<const float *>(&block.fft_output));

    fft_seq_dataset_.resize({n + 1, 2});
    std::vector<size_t> seq_nums = {block.start_seq_num, block.end_seq_num};
    fft_seq_dataset_.select({n, 0}, {1, 2}).write_raw(seq_nums.data());
    ++n_blocks_written_;

    // auto missing_size = fft_missing_dataset_.getDimensions()[0];
    // fft_missing_dataset_.resize({missing_size + 1, 3});
    // float num_missing_packets_fl = static_cast<float>(num_missing_packets);
    // float num_total_packets_fl = static_cast<float>(num_total_packets);
    // std::vector<float> missing_nums = {
    //     num_missing_packets_fl, num_total_packets_fl,
    //     100 * num_missing_packets_fl / num_total_packets_fl};
    // fft_missing_dataset_.select({missing_size, 0}, {1, 3})
    //     .write_raw(missing_nums.data());
  }

  void flush() override { file_.flush(); }

private:
  HighFive::File &file_;
  size_t element_count_;
  HighFive::DataSet fft_dataset_;
  HighFive::DataSet fft_seq_dataset_;
  std::vector<size_t> fft_dims_;
  size_t n_blocks_written_ = 0;

  size_t batch_size_;
  size_t buffer_count_ = 0;
  size_t fft_elements_ = 1;

  std::vector<float> fft_buffer_;
  std::vector<size_t> seq_buffer_;
};

template <typename T>
class HDF5VisibilitiesWriter : public VisibilitiesWriter<T> {
public:
  // deflate_level: 0 = no compression (fast, recommended for real-time writes),
  //               1-9 = zlib level (smaller files, slower).
  HDF5VisibilitiesWriter(
      HighFive::File &file, const int min_channel, const int max_channel,
      const std::unordered_map<int, int> *antenna_map = nullptr,
      const int num_blocks = 100, const int deflate_level = 0)
      : VisibilitiesWriter<T>(num_blocks), file_(file), batch_size_(num_blocks),
        element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)) {

    std::cout << "HDF5Visibilities Writer starting with element_count "
              << element_count_ << " and total size " << sizeof(T) << " bytes."
              << " Writing in batch_size " << batch_size_ << std::endl;

    vis_buffer_.reserve(batch_size_ * element_count_);
    missing_buffer_.reserve(batch_size_ * 3);
    seq_buffer_.reserve(batch_size_ * 2);

    using namespace HighFive;

    double start_time = std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch())
                                .count() /
                            86400.0 +
                        40587.0;
    file_.createAttribute<double>("mjd_start", start_time);
    file_.createAttribute<int>("min_channel", min_channel);
    file_.createAttribute<int>("max_channel", max_channel);

    vis_dims_ = get_array_dims<T>();
    std::vector<size_t> vis_dataset_dims = {0};
    std::vector<size_t> vis_dataset_max_dims = {DataSpace::UNLIMITED};

    vis_dataset_dims.insert(vis_dataset_dims.end(), vis_dims_.begin(),
                            vis_dims_.end());
    vis_dataset_max_dims.insert(vis_dataset_max_dims.end(), vis_dims_.begin(),
                                vis_dims_.end());
    std::vector<hsize_t> vis_chunk = {static_cast<hsize_t>(batch_size_)};
    vis_chunk.insert(vis_chunk.end(), vis_dims_.begin(), vis_dims_.end());
    DataSpace vis_space(vis_dataset_dims, vis_dataset_max_dims);
    DataSetCreateProps props;
    props.add(HighFive::Chunking(vis_chunk));
    if (deflate_level > 0) {
      props.add(HighFive::Shuffle());
      props.add(HighFive::Deflate(deflate_level));
    }
    vis_dataset_ = file_.createDataSet<float>("visibilities", vis_space, props);

    DataSetCreateProps vis_seq_props;
    vis_seq_props.add(Chunking(std::vector<hsize_t>{batch_size_, 2}));
    vis_seq_dataset_ = file_.createDataSet<size_t>(
        "vis_seq_nums", DataSpace({0, 2}, {HighFive::DataSpace::UNLIMITED, 2}),
        vis_seq_props);

    DataSetCreateProps vis_missing_props;
    vis_missing_props.add(Chunking(std::vector<hsize_t>{batch_size_, 3}));
    vis_missing_dataset_ = file_.createDataSet<float>(
        "vis_missing_nums",
        DataSpace({0, 3}, {HighFive::DataSpace::UNLIMITED, 3}),
        vis_missing_props);

    if (antenna_map && !antenna_map->empty()) {
      this->antenna_map_ = *antenna_map;
    } else {
      generate_identity_map();
    }

    write_baseline_ids();
  }

  void process_block(const VisBlock<T> &block) override {
    const float *vis_ptr = reinterpret_cast<const float *>(&block.data);
    vis_buffer_.insert(vis_buffer_.end(), vis_ptr, vis_ptr + element_count_);

    seq_buffer_.push_back(block.start_seq_num);
    seq_buffer_.push_back(block.end_seq_num);

    float num_missing_packets_fl =
        static_cast<float>(block.num_missing_packets);
    float num_total_packets_fl = static_cast<float>(block.num_total_packets);

    missing_buffer_.push_back(num_missing_packets_fl);
    missing_buffer_.push_back(num_total_packets_fl);
    missing_buffer_.push_back(100 * num_missing_packets_fl /
                              num_total_packets_fl);

    buffer_count_++;

    if (buffer_count_ >= batch_size_) {
      flush();
    }
  }

  void flush() override {
    if (buffer_count_ == 0)
      return;
    // Use locally tracked size to avoid a getDimensions() HDF5 metadata round-
    // trip on every flush — confirmed safe because only this thread appends.
    const size_t current_size = n_blocks_written_;
    std::vector<size_t> new_dims = {current_size + buffer_count_};
    new_dims.insert(new_dims.end(), vis_dims_.begin(), vis_dims_.end());
    vis_dataset_.resize(new_dims);

    std::vector<size_t> vis_offset = {current_size};
    vis_offset.insert(vis_offset.end(), vis_dims_.size(), 0);
    std::vector<size_t> vis_count = {buffer_count_};
    vis_count.insert(vis_count.end(), vis_dims_.begin(), vis_dims_.end());

    vis_dataset_.select(vis_offset, vis_count).write_raw(vis_buffer_.data());

    vis_seq_dataset_.resize({current_size + buffer_count_, 2});
    vis_seq_dataset_.select({current_size, 0}, {buffer_count_, 2})
        .write_raw(seq_buffer_.data());

    vis_missing_dataset_.resize({current_size + buffer_count_, 3});
    vis_missing_dataset_.select({current_size, 0}, {buffer_count_, 3})
        .write_raw(missing_buffer_.data());

    n_blocks_written_ += buffer_count_;
    vis_buffer_.clear();
    seq_buffer_.clear();
    missing_buffer_.clear();
    buffer_count_ = 0;
  }

private:
  void write_baseline_ids() {
    // Extract NR_BASELINES from T.
    // T is float[CH][BL][POL][POL][CPLX].
    // extent<T, 0> is Channels, extent<T, 1> is Baselines.
    constexpr size_t nr_baselines = std::extent<T, 1>::value;

    // Calculate number of antennas from triangular number formula:
    // B = A(A+1)/2  =>  A^2 + A - 2B = 0
    // A = (-1 + sqrt(1 + 8B)) / 2
    size_t nr_antennas =
        static_cast<size_t>((std::sqrt(1 + 8 * nr_baselines) - 1) / 2);

    std::vector<int> baseline_ids;
    baseline_ids.reserve(nr_baselines);

    // Generate FITS IDs (256 * ant1 + ant2) based on triangular order
    // Order: 0-0, 0-1, 1-1, 0-2, 1-2, 2-2 ...
    for (size_t ant2 = 0; ant2 < nr_antennas; ++ant2) {
      for (size_t ant1 = 0; ant1 <= ant2; ++ant1) {
        baseline_ids.push_back(256 * antenna_map_[ant1] + antenna_map_[ant2]);
      }
    }

    // Sanity check
    if (baseline_ids.size() != nr_baselines) {
      // Handle error/log warning here if dimensions don't match expectation
    }

    // Write to HDF5 (Static dataset, no need for Unlimited/Chunking)
    file_
        .createDataSet<int>("baseline_ids",
                            HighFive::DataSpace::From(baseline_ids))
        .write(baseline_ids);
  }

  void generate_identity_map() {
    for (int i = 0; i < 256; i++) {
      antenna_map_[i] = i;
    }
  }

  HighFive::File &file_;
  size_t element_count_;
  HighFive::DataSet vis_dataset_;
  HighFive::DataSet vis_seq_dataset_;
  HighFive::DataSet vis_missing_dataset_;
  std::vector<size_t> vis_dims_;
  std::unordered_map<int, int> antenna_map_;

  size_t batch_size_;
  size_t buffer_count_ = 0;
  size_t n_blocks_written_ = 0;
  size_t vis_elements_ = 1;

  std::vector<float> vis_buffer_;
  std::vector<float> missing_buffer_;
  std::vector<size_t> seq_buffer_;
};

template <typename TVal, typename TVec>
class RedisEigendataWriter : public EigenWriter<TVal, TVec> {
public:
  // channels_per_write: how many channels to write per output block.
  // 0 means write all channels every block (original behaviour).
  // Values < NR_CHANNELS enable round-robin to cap per-block Redis payload.
  RedisEigendataWriter(const int channels_per_write = 0,
                       const int num_blocks = 100)
      : EigenWriter<TVal, TVec>(num_blocks),
        val_element_count_(
            sizeof(TVal) /
            sizeof(typename std::remove_all_extents<TVal>::type)),
        vec_element_count_(
            sizeof(TVec) /
            sizeof(typename std::remove_all_extents<TVec>::type)),
        redis("tcp://127.0.0.1:6379") {

    eigen_dims_ = get_array_dims<TVal>();

    NR_CHANNELS = eigen_dims_[0];
    NR_POLARIZATIONS = eigen_dims_[1];
    NR_RECEIVERS = eigen_dims_[3];

    channels_per_write_ = (channels_per_write <= 0)
                              ? NR_CHANNELS
                              : std::min(channels_per_write, NR_CHANNELS);
    num_slots_ = (NR_CHANNELS + channels_per_write_ - 1) / channels_per_write_;
    current_slot_ = 0;

    std::cout << "RedisEigendataWriter initialized with NR_CHANNELS: "
              << NR_CHANNELS << ", NR_POL: " << NR_POLARIZATIONS
              << ", NR_RECEIVERS: " << NR_RECEIVERS
              << ", channels_per_write: " << channels_per_write_
              << ", num_slots: " << num_slots_ << std::endl;

    create_all_timeseries_keys();
    preallocate_madd_args();
  }

  void process_block(const EigenBlock<TVal, TVec> &block) override {
    long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    std::string ts_str = std::to_string(ts);

    const int N = NR_RECEIVERS;
    const float *val_ptr = reinterpret_cast<const float *>(&block.eigenvalues);
    const std::complex<float> *vec_ptr =
        reinterpret_cast<const std::complex<float> *>(&block.eigenvectors);

    auto update_val = [](std::string &str, float val) {
      str.resize(32);
      auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), val);
      str.resize(ptr - str.data());
    };

    auto &args = slot_madd_args_[current_slot_];
    const int ch_start = current_slot_ * channels_per_write_;
    const int ch_end = std::min(ch_start + channels_per_write_, NR_CHANNELS);

    size_t arg_idx = 0;

    for (int ch_idx = ch_start; ch_idx < ch_end; ++ch_idx) {
      // Loop only over auto-polarizations (0-0, 1-1, etc.)
      for (int pol_idx = 0; pol_idx < NR_POLARIZATIONS; ++pol_idx) {
        // Offset math uses global ch_idx because the memory array spans all
        // channels; we are just writing a subset of channels this block.
        size_t val_base_offset =
            ch_idx * (NR_POLARIZATIONS * NR_POLARIZATIONS * N) +
            pol_idx * (NR_POLARIZATIONS * N) + pol_idx * N;

        size_t vec_base_offset =
            ch_idx * (NR_POLARIZATIONS * NR_POLARIZATIONS * N * N) +
            pol_idx * (NR_POLARIZATIONS * N * N) + pol_idx * (N * N);

        for (int k_idx = 0; k_idx < N; ++k_idx) {

          // --- Eigenvalue ---
          const float eigenvalue = val_ptr[val_base_offset + k_idx];
          args[1 + arg_idx * 3 + 1] = ts_str;
          update_val(args[1 + arg_idx * 3 + 2], eigenvalue);
          arg_idx++;

          // --- Eigenvectors ---
          size_t vec_k_offset = vec_base_offset + k_idx * N;
          for (int j_idx = 0; j_idx < N; ++j_idx) {
            const std::complex<float> &coeff = vec_ptr[vec_k_offset + j_idx];

            args[1 + arg_idx * 3 + 1] = ts_str;
            update_val(args[1 + arg_idx * 3 + 2], coeff.real());
            arg_idx++;

            args[1 + arg_idx * 3 + 1] = ts_str;
            update_val(args[1 + arg_idx * 3 + 2], coeff.imag());
            arg_idx++;
          }
        }
      }
    }

    if (args.size() > 1) {
      redis.command(args.begin(), args.end());
    }

    current_slot_ = (current_slot_ + 1) % num_slots_;
  }

  void flush() override {}

private:
  size_t val_element_count_;
  size_t vec_element_count_;
  std::vector<size_t> eigen_dims_;
  sw::redis::Redis redis;
  int NR_CHANNELS;
  int NR_POLARIZATIONS;
  int NR_RECEIVERS;

  int channels_per_write_;
  int num_slots_;
  int current_slot_;
  std::vector<std::vector<std::string>> slot_madd_args_;

  void preallocate_madd_args() {
    const int N = NR_RECEIVERS;
    const size_t data_points_per_k = 1 + (N * 2);

    slot_madd_args_.resize(num_slots_);

    for (int s = 0; s < num_slots_; ++s) {
      const int ch_start = s * channels_per_write_;
      const int ch_end = std::min(ch_start + channels_per_write_, NR_CHANNELS);
      const int channels_in_slot = ch_end - ch_start;
      const size_t total_data_points =
          channels_in_slot * NR_POLARIZATIONS * N * data_points_per_k;

      auto &args = slot_madd_args_[s];
      args.resize(1 + total_data_points * 3);
      args[0] = "TS.MADD";

      size_t arg_idx = 0;
      for (int ch_idx = ch_start; ch_idx < ch_end; ++ch_idx) {
        std::string channel_id = std::to_string(ch_idx);

        for (int pol_idx = 0; pol_idx < NR_POLARIZATIONS; ++pol_idx) {
          std::string pol_pair =
              std::to_string(pol_idx) + "-" + std::to_string(pol_idx);

          for (int k_idx = 0; k_idx < N; ++k_idx) {
            std::string k_id = std::to_string(k_idx);
            std::string key_prefix =
                "ts:ch:" + channel_id + ":p:" + pol_pair + ":k:" + k_id;

            args[1 + arg_idx * 3] = key_prefix + ":val";
            arg_idx++;

            for (int j_idx = 0; j_idx < N; ++j_idx) {
              std::string vec_key_prefix =
                  key_prefix + ":vec:j:" + std::to_string(j_idx);

              args[1 + arg_idx * 3] = vec_key_prefix + ":re";
              arg_idx++;

              args[1 + arg_idx * 3] = vec_key_prefix + ":im";
              arg_idx++;
            }
          }
        }
      }
    }
  }

  void create_all_timeseries_keys() {
    const int N = NR_RECEIVERS;
    const std::vector<std::string> val_components = {"val"};
    const std::vector<std::string> vec_components = {"re", "im"};

    std::cout << "Starting TimeSeries key pre-creation..." << std::endl;

    // We only create auto-polarization keys (total pairs = NR_POLARIZATIONS)
    int total_pol_pairs = NR_POLARIZATIONS;
    int total_eigenvalue_keys =
        NR_CHANNELS * total_pol_pairs * N * val_components.size();
    int total_eigenvector_keys =
        NR_CHANNELS * total_pol_pairs * N * vec_components.size() * N;

    std::cout << "Total keys to create: "
              << (total_eigenvalue_keys + total_eigenvector_keys) << std::endl;

    for (int ch_idx = 0; ch_idx < NR_CHANNELS; ++ch_idx) {
      std::string channel_id = std::to_string(ch_idx);

      // Loop only over auto-polarizations
      for (int pol_idx = 0; pol_idx < NR_POLARIZATIONS; ++pol_idx) {
        std::string pol_pair =
            std::to_string(pol_idx) + "-" + std::to_string(pol_idx);

        for (int k_idx = 0; k_idx < N; ++k_idx) {
          std::string k_id = std::to_string(k_idx);

          for (const auto &component : val_components) {
            std::string key = "ts:ch:" + channel_id + ":p:" + pol_pair +
                              ":k:" + k_id + ":" + component;
            std::vector<std::string> args = {
                "TS.CREATE",    key,      "LABELS",    "channel", channel_id,
                "polarization", pol_pair, "eigen_idx", k_id,      "component",
                component};
            try {
              redis.command(args.begin(), args.end());
            } catch (const std::exception &e) {
              ERROR_LOG("Error creating key {}: {}", key, e.what());
            }
          }

          for (int j_idx = 0; j_idx < N; ++j_idx) {
            std::string j_id = std::to_string(j_idx);
            for (const auto &component : vec_components) {
              std::string key = "ts:ch:" + channel_id + ":p:" + pol_pair +
                                ":k:" + k_id + ":vec:j:" + j_id + ":" +
                                component;
              std::vector<std::string> args = {
                  "TS.CREATE", key,         "LABELS",
                  "channel",   channel_id,  "polarization",
                  pol_pair,    "eigen_idx", k_id,
                  "vec_idx",   j_id,        "component",
                  component};
              try {
                redis.command(args.begin(), args.end());
              } catch (const std::exception &e) {
                ERROR_LOG("Error creating key {}: {}", key, e.what());
              }
            }
          }
        }
      }
    }
    std::cout << "TimeSeries key pre-creation complete." << std::endl;
  }
};

template <typename T> class RedisBeamFFTWriter : public FFTWriter<T> {
public:
  // channels_per_write: how many channels to write per output block.
  // 0 means write all channels every block (original behaviour).
  // Values < NR_CHANNELS enable round-robin to cap per-block Redis payload.
  RedisBeamFFTWriter(int num_channels, int num_beams, int num_polarizations,
                     std::string prefix = "", const int num_blocks = 100,
                     const int channels_per_write = 0)
      : FFTWriter<T>(num_blocks),
        element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)),
        redis("tcp://127.0.0.1:6379"), NR_CHANNELS(num_channels),
        NR_BEAMS(num_beams), NR_POLARIZATIONS(num_polarizations),
        prefix(prefix) {
    fft_dims_ = get_array_dims<T>();
    NR_FREQS = fft_dims_[fft_dims_.size() - 1];

    channels_per_write_ = (channels_per_write <= 0)
                              ? NR_CHANNELS
                              : std::min(channels_per_write, NR_CHANNELS);
    num_slots_ = (NR_CHANNELS + channels_per_write_ - 1) / channels_per_write_;
    current_slot_ = 0;

    std::cout << "FFT Dims are ";
    for (auto dim : fft_dims_) {
      std::cout << dim << ", ";
    }
    std::cout << std::endl;

    precomputed_keys.resize(NR_CHANNELS * NR_POLARIZATIONS * NR_FREQS *
                            NR_BEAMS);
    precomputed_max_keys.resize(NR_CHANNELS * NR_POLARIZATIONS * NR_FREQS *
                                NR_BEAMS);
    precomputed_max_100ms_keys.resize(NR_CHANNELS * NR_POLARIZATIONS *
                                      NR_FREQS * NR_BEAMS);

    for (int ch = 0; ch < NR_CHANNELS; ++ch) {
      for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
        for (int beam = 0; beam < NR_BEAMS; beam++) {
          for (int f = 0; f < NR_FREQS; ++f) {
            // Apply the frequency shift (F/2) during pre-caching
            // so we don't calculate it during the hot loop.
            int f_shifted = (f + NR_FREQS / 2) % NR_FREQS;

            std::string key = prefix + "ts:fft:ch:" + std::to_string(ch) +
                              ":p:" + std::to_string(pol) +
                              ":b:" + std::to_string(beam) +
                              ":f:" + std::to_string(f_shifted);

            std::string max_key = "ts:fft_max1s:ch:" + std::to_string(ch) +
                                  ":p:" + std::to_string(pol) +
                                  ":b:" + std::to_string(beam) +
                                  ":f:" + std::to_string(f_shifted);

            std::string max_100ms_key =
                "ts:fft_max100ms:ch:" + std::to_string(ch) +
                ":p:" + std::to_string(pol) + ":b:" + std::to_string(beam) +
                ":f:" + std::to_string(f_shifted);
            precomputed_keys[get_key_index(ch, pol, beam, f)] = key;
            precomputed_max_keys[get_key_index(ch, pol, beam, f)] = max_key;
            precomputed_max_100ms_keys[get_key_index(ch, pol, beam, f)] =
                max_100ms_key;
          }
        }
      }
    }

    std::cout << "RedisBeamFFTWriter has NR_CHANNELS: " << NR_CHANNELS
              << ", NR_BEAMS:" << NR_BEAMS << ", NR_FREQS: " << NR_FREQS
              << ", NR_POLARIZATIONS: " << NR_POLARIZATIONS
              << ", channels_per_write: " << channels_per_write_
              << ", num_slots: " << num_slots_ << std::endl;
    create_all_timeseries_keys();

    // Pre-allocate one madd_args vector per slot with keys pre-seeded.
    slot_madd_args_.resize(num_slots_);
    for (int s = 0; s < num_slots_; ++s) {
      const int ch_start = s * channels_per_write_;
      const int ch_end = std::min(ch_start + channels_per_write_, NR_CHANNELS);
      const int channels_in_slot = ch_end - ch_start;
      auto &args = slot_madd_args_[s];
      args.resize(1 + channels_in_slot * NR_POLARIZATIONS * NR_BEAMS *
                          NR_FREQS * 3);
      args[0] = "TS.MADD";
      int idx = 0;
      for (int ch = ch_start; ch < ch_end; ++ch) {
        for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
          for (int beam = 0; beam < NR_BEAMS; ++beam) {
            for (int f = 0; f < NR_FREQS; ++f) {
              args[1 + 3 * idx] =
                  precomputed_keys[get_key_index(ch, pol, beam, f)];
              idx++;
            }
          }
        }
      }
    }
  }

  void process_block(const FFTBlock<T> &block) override {
    long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();
    std::string ts_str = std::to_string(ts);

    auto &args = slot_madd_args_[current_slot_];
    const int ch_start = current_slot_ * channels_per_write_;
    const int ch_end = std::min(ch_start + channels_per_write_, NR_CHANNELS);

    int current_idx = 0;
    for (int ch = ch_start; ch < ch_end; ++ch) {
      for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
        for (int beam = 0; beam < NR_BEAMS; ++beam) {
          for (int f = 0; f < NR_FREQS; ++f) {
            const auto cval = block.fft_output[ch][pol][beam][f];
            args[1 + 3 * current_idx + 1] = ts_str;

            std::string &val_str = args[1 + 3 * current_idx + 2];
            val_str.resize(32);
            auto [ptr, ec] = std::to_chars(
                val_str.data(), val_str.data() + val_str.size(), cval);
            val_str.resize(ptr - val_str.data());
            current_idx += 1;
          }
        }
      }
    }
    redis.command(args.begin(), args.end());
    current_slot_ = (current_slot_ + 1) % num_slots_;
  }

  void flush() override {}

private:
  inline size_t get_key_index(int ch, int pol, int beam, int f) const {
    return static_cast<size_t>(ch) * (NR_POLARIZATIONS * NR_FREQS * NR_BEAMS) +
           static_cast<size_t>(pol) * (NR_FREQS * NR_BEAMS) +
           static_cast<size_t>(beam) * NR_FREQS + static_cast<size_t>(f);
  }
  std::string prefix;

  int channels_per_write_;
  int num_slots_;
  int current_slot_;
  std::vector<std::vector<std::string>> slot_madd_args_;
  void create_all_timeseries_keys() {
    std::cout << "Pre-creating FFT TimeSeries keys..." << std::endl;
    for (int ch = 0; ch < NR_CHANNELS; ++ch) {
      for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
        for (int beam = 0; beam < NR_BEAMS; ++beam) {
          for (int f = 0; f < NR_FREQS; ++f) {
            int f_shifted = (f + NR_FREQS / 2) % NR_FREQS;
            auto key = precomputed_keys[get_key_index(ch, pol, beam, f)];
            auto max_key =
                precomputed_max_keys[get_key_index(ch, pol, beam, f)];
            auto max_100ms_key =
                precomputed_max_100ms_keys[get_key_index(ch, pol, beam, f)];

            std::vector<std::string> args = {"TS.CREATE",
                                             key,
                                             "RETENTION",
                                             "60000",
                                             "LABELS",
                                             "type",
                                             "raw",
                                             "channel",
                                             std::to_string(ch),
                                             "polarization",
                                             std::to_string(pol),
                                             "freq",
                                             std::to_string(f_shifted),
                                             "beam",
                                             std::to_string(beam),
                                             "component",
                                             "beam-bandpass"};

            std::vector<std::string> max_args = {"TS.CREATE",
                                                 max_key,
                                                 "RETENTION",
                                                 "0",
                                                 "LABELS",
                                                 "type",
                                                 "aggregated",
                                                 "channel",
                                                 std::to_string(ch),
                                                 "polarization",
                                                 std::to_string(pol),
                                                 "freq",
                                                 std::to_string(f_shifted),
                                                 "beam",
                                                 std::to_string(beam),
                                                 "component",
                                                 "beam-bandpass_max1s"};

            std::vector<std::string> max_100ms_args = {
                "TS.CREATE",
                max_100ms_key,
                "RETENTION",
                "0",
                "LABELS",
                "type",
                "aggregated",
                "channel",
                std::to_string(ch),
                "polarization",
                std::to_string(pol),
                "freq",
                std::to_string(f_shifted),
                "beam",
                std::to_string(beam),
                "component",
                "beam-bandpass_max100ms"};
            std::vector<std::string> rule_args = {
                "TS.CREATERULE", key, max_key, "AGGREGATION", "max", "1000"};
            std::vector<std::string> rule_100ms_args = {
                "TS.CREATERULE", key,   max_100ms_key,
                "AGGREGATION",   "max", "100"};
            try {
              redis.command(args.begin(), args.end());
              redis.command(max_args.begin(), max_args.end());
              redis.command(max_100ms_args.begin(), max_100ms_args.end());
              redis.command(rule_args.begin(), rule_args.end());
              redis.command(rule_100ms_args.begin(), rule_100ms_args.end());
            } catch (const std::exception &e) {
              ERROR_LOG("Error creating key {}: {}", key, e.what());
            }
          }
        }
      }
    }

    std::cout << "FFT TimeSeries key creation complete." << std::endl;
  };
  size_t element_count_;
  std::vector<size_t> fft_dims_;
  sw::redis::Redis redis;
  int NR_CHANNELS;
  int NR_POLARIZATIONS;
  int NR_BEAMS;
  int NR_FREQS;
  std::vector<std::string> precomputed_keys;
  std::vector<std::string> precomputed_max_keys;
  std::vector<std::string> precomputed_max_100ms_keys;
};

template <typename T, size_t N> constexpr size_t array_dimension() { return N; }

template <typename T> hid_t get_hdf5_type() {
  if constexpr (std::is_same_v<T, float>)
    return H5T_NATIVE_FLOAT;
  else if constexpr (std::is_same_v<T, double>)
    return H5T_NATIVE_DOUBLE;
  else if constexpr (std::is_same_v<T, int>)
    return H5T_NATIVE_INT;
  else if constexpr (std::is_same_v<T, unsigned int>)
    return H5T_NATIVE_UINT;
  else if constexpr (std::is_same_v<T, short>)
    return H5T_NATIVE_SHORT;
  else if constexpr (std::is_same_v<T, unsigned short>)
    return H5T_NATIVE_USHORT;
  else if constexpr (std::is_same_v<T, char>)
    return H5T_NATIVE_CHAR;
  else if constexpr (std::is_same_v<T, unsigned char>)
    return H5T_NATIVE_UCHAR;
  else if constexpr (std::is_same_v<T, bool>)
    return H5T_NATIVE_HBOOL;
  else
    return H5T_NATIVE_INT; // fallback
}

template <typename BeamT, typename ArrivalsT>
class PSRDADABeamWriter : public BeamWriter<BeamT, ArrivalsT> {
public:
  PSRDADABeamWriter(std::string header_filename, const int num_blocks = 100)
      : BeamWriter<BeamT, ArrivalsT>(num_blocks) {

    log = 0;

    // default shared memory key
    key_t dada_key = DADA_DEFAULT_BLOCK_KEY;

    // DADA Header + Data unit
    hdu = 0;

    header_file = strdup(header_filename.c_str());
    obs_header = (char *)malloc(sizeof(char) * DADA_DEFAULT_HEADER_SIZE);
    if (!obs_header) {
      fprintf(stderr, "ERROR: could not allocate memory\n");
      throw std::runtime_error();
    }

    // read the ASCII DADA header from the file
    if (fileread(header_file, obs_header, DADA_DEFAULT_HEADER_SIZE) < 0) {
      free(obs_header);
      fprintf(stderr, "ERROR: could not read ASCII header from %s\n",
              header_file);
      throw std::runtime_error();
    }

    // create a multilogger
    log = multilog_open("PSRDADABeamWriter", 0);

    // set the destination for multilog to stderr
    multilog_add(log, stderr);

    // create the HDU struct
    hdu = dada_hdu_create(log);

    // set the key to connecting to the HDU
    dada_hdu_set_key(hdu, dada_key);

    // connect to HDU
    if (dada_hdu_connect(hdu) < 0) {
      multilog(log, LOG_ERR, "could not connect to HDU\n");

      throw std::runtime_error();
    }

    if (dada_hdu_lock_write(hdu) < 0) {
      multilog(log, LOG_ERR, "could not lock write on HDU\n");
      throw std::runtime_error();
    }
  }

  ~PSRDADABeamWriter() {

    if (dada_hdu_unlock_write(hdu) < 0) {
      multilog(log, LOG_ERR, "dada_hdu_unlock_write failed\n");
      throw std::runtime_error();
    }

    // disconnect from HDU
    if (dada_hdu_disconnect(hdu) < 0)
      multilog(log, LOG_ERR, "could not unlock write on hdu\n");
  }

  void process_block(const BeamBlock<BeamT, ArrivalsT> &block) {
    {
      uint64_t header_size = ipcbuf_get_bufsz(hdu->header_block);
      char *header = ipcbuf_get_next_write(hdu->header_block);
      memcpy(header, obs_header, header_size);

      // Enable EOD so that subsequent transfers will move to the next buffer in
      // the header block
      if (ipcbuf_enable_eod(hdu->header_block) < 0) {
        multilog(log, LOG_ERR, "Could not enable EOD on Header Block\n");
        throw std::runtime_error();
      }

      // flag the header block for this "observation" as filled
      if (ipcbuf_mark_filled(hdu->header_block, header_size) < 0) {
        multilog(log, LOG_ERR, "could not mark filled Header Block\n");
        throw std::runtime_error();
      }
    }

    // the size of 1 block (buffer element) in the data block
    uint64_t block_size = ipcbuf_get_bufsz((ipcbuf_t *)hdu->data_block);

    // write 1 block worth of data block via the "block" method
    {
      uint64_t block_id;
      char *ring_block = ipcio_open_block_write(hdu->data_block, &block_id);
      if (!block) {
        multilog(log, LOG_ERR, "ipcio_open_block_write failed\n");
        throw std::runtime_error();
      }

      // this is where I should copy in the data
      memset(ring_block, 0, block_size);

      if (ipcio_close_block_write(hdu->data_block, block_size) < 0) {
        multilog(log, LOG_ERR, "ipcio_close_block_write failed\n");
        throw std::runtime_error();
      }
    }
  }

  void flush() {}

private:
  multilog_t *log;
  key_t data_key;
  dada_hdu_t *hdu;
  char *obs_header;
  char *header_file;
};

template <typename TVal, typename TVec>
class HDF5ProjectionEigenWriter : public EigenWriter<TVal, TVec> {
public:
  explicit HDF5ProjectionEigenWriter(HighFive::File &file,
                                     const int num_blocks = 100)
      : EigenWriter<TVal, TVec>(num_blocks), file_(file) {
    using namespace HighFive;

    vec_dims_ = get_array_dims<TVec>();
    vec_dims_.push_back(2); // add the complex as we will be saving out as float
                            // but input is likely std::complex<float>

    // Outer (block) dimension is unlimited; inner shape comes from TVec.
    std::vector<size_t> dataset_dims = {0};
    std::vector<size_t> dataset_max_dims = {DataSpace::UNLIMITED};
    dataset_dims.insert(dataset_dims.end(), vec_dims_.begin(), vec_dims_.end());
    dataset_max_dims.insert(dataset_max_dims.end(), vec_dims_.begin(),
                            vec_dims_.end());

    // Chunk: one block at a time — matches the write pattern.
    std::vector<hsize_t> chunk = {1};
    chunk.insert(chunk.end(), vec_dims_.begin(), vec_dims_.end());

    DataSetCreateProps vec_props;
    vec_props.add(Chunking(chunk));
    vec_props.add(Deflate(4));

    // Store as float32 (complex<float> is two consecutive floats in memory).
    vec_dataset_ = file_.createDataSet<float>(
        "projection_eigenvectors", DataSpace(dataset_dims, dataset_max_dims),
        vec_props);

    using namespace HighFive;

    val_dims_ = get_array_dims<TVal>();

    // Outer (block) dimension is unlimited; inner shape comes from TVec.
    std::vector<size_t> dataset_dims_vals = {0};
    std::vector<size_t> dataset_max_dims_vals = {DataSpace::UNLIMITED};
    dataset_dims_vals.insert(dataset_dims_vals.end(), val_dims_.begin(),
                             val_dims_.end());
    dataset_max_dims_vals.insert(dataset_max_dims_vals.end(), val_dims_.begin(),
                                 val_dims_.end());

    // Chunk: one block at a time — matches the write pattern.
    std::vector<hsize_t> chunk_vals = {1};
    chunk_vals.insert(chunk_vals.end(), val_dims_.begin(), val_dims_.end());

    DataSetCreateProps val_props;
    val_props.add(Chunking(chunk_vals));
    val_props.add(Deflate(4));

    // Store as float32 (complex<float> is two consecutive floats in memory).
    val_dataset_ = file_.createDataSet<float>(
        "projection_eigenvalues",
        DataSpace(dataset_dims_vals, dataset_max_dims_vals), val_props);

    // Sequence number dataset: [N_blocks, 2].
    DataSetCreateProps seq_props;
    seq_props.add(Chunking(std::vector<hsize_t>{1, 2}));
    seq_dataset_ = file_.createDataSet<size_t>(
        "projection_seq_nums", DataSpace({0, 2}, {DataSpace::UNLIMITED, 2}),
        seq_props);
  }

  // -------------------------------------------------------------------------
  // write_eigendata_block
  //
  // val_data — always nullptr for this pipeline, ignored.
  // vec_data — pointer to one TVec worth of eigenvector data.
  // -------------------------------------------------------------------------
  void process_block(const EigenBlock<TVal, TVec> &block) override {
    const size_t n = n_blocks_written_;

    // ---- eigenvalues -------------------------------------------------------
    std::vector<size_t> val_dims = {n + 1};
    val_dims.insert(val_dims.end(), val_dims_.begin(), val_dims_.end());
    val_dataset_.resize(val_dims);

    std::vector<size_t> val_offset = {n};
    val_offset.insert(val_offset.end(), val_dims_.size(), 0);
    std::vector<size_t> val_count = {1};
    val_count.insert(val_count.end(), val_dims_.begin(), val_dims_.end());
    val_dataset_.select(val_offset, val_count).write_raw(&block.eigenvalues[0]);

    // ---- eigenvectors -------------------------------------------------------
    std::vector<size_t> vec_dims = {n + 1};
    vec_dims.insert(vec_dims.end(), vec_dims_.begin(), vec_dims_.end());
    vec_dataset_.resize(vec_dims);

    std::vector<size_t> vec_offset = {n};
    vec_offset.insert(vec_offset.end(), vec_dims_.size(), 0);
    std::vector<size_t> vec_count = {1};
    vec_count.insert(vec_count.end(), vec_dims_.begin(), vec_dims_.end());
    vec_dataset_.select(vec_offset, vec_count)
        .write_raw(&block.eigenvectors[0]);

    // ---- sequence numbers ---------------------------------------------------
    seq_dataset_.resize({n + 1, 2});
    const std::vector<size_t> seq_nums = {block.start_seq_num,
                                          block.end_seq_num};
    seq_dataset_.select({n, 0}, {1, 2}).write_raw(seq_nums.data());

    ++n_blocks_written_;
  }

  void flush() override { file_.flush(); }

private:
  HighFive::File &file_;
  HighFive::DataSet vec_dataset_;
  HighFive::DataSet val_dataset_;
  HighFive::DataSet seq_dataset_;
  std::vector<size_t> vec_dims_; // inner shape of TVec, e.g. {CH,POL,POL,N,N}
  std::vector<size_t> val_dims_;
  size_t n_blocks_written_ = 0;
};

template <typename T> class HDF5BeamFFTWriter : public FFTWriter<T> {
public:
  HDF5BeamFFTWriter(HighFive::File &file, const int min_channel,
                    const int max_channel,
                    const std::unordered_map<int, int> *antenna_map = nullptr,
                    const int num_blocks = 100)
      : FFTWriter<T>(num_blocks), file_(file), batch_size_(num_blocks),
        element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)) {
    using namespace HighFive;

    fft_buffer_.reserve(batch_size_ * element_count_);
    seq_buffer_.reserve(batch_size_ * 2);

    double start_time = std::chrono::duration<double>(
                            std::chrono::system_clock::now().time_since_epoch())
                                .count() /
                            86400.0 +
                        40587.0;
    file_.createAttribute<double>("mjd_start", start_time);
    file_.createAttribute<int>("min_channel", min_channel);
    file_.createAttribute<int>("max_channel", max_channel);

    fft_dims_ = get_array_dims<T>();
    std::cout << "HDF5BeamFFTWriter initializing with dimensions (";
    for (auto _i : fft_dims_) {
      std::cout << _i << ", ";
    }
    std::cout << ").\n";

    // Dataset dims: [0, ...fft_dims_], unlimited on first axis
    std::vector<size_t> fft_dataset_dims = {0};
    std::vector<size_t> fft_dataset_max_dims = {DataSpace::UNLIMITED};
    fft_dataset_dims.insert(fft_dataset_dims.end(), fft_dims_.begin(),
                            fft_dims_.end());
    fft_dataset_max_dims.insert(fft_dataset_max_dims.end(), fft_dims_.begin(),
                                fft_dims_.end());

    // Chunk: 1 block at a time
    std::vector<hsize_t> fft_chunk = {static_cast<hsize_t>(batch_size_)};
    fft_chunk.insert(fft_chunk.end(), fft_dims_.begin(), fft_dims_.end());

    DataSpace fft_space(fft_dataset_dims, fft_dataset_max_dims);
    DataSetCreateProps props;
    props.add(Chunking(fft_chunk));
    props.add(Shuffle());
    props.add(Deflate(1));
    fft_dataset_ = file_.createDataSet<float>("beam_ffts", fft_space, props);

    // Sequence number dataset: [N, 2] (start_seq, end_seq)
    DataSetCreateProps fft_seq_props;
    fft_seq_props.add(
        Chunking(std::vector<hsize_t>{static_cast<hsize_t>(batch_size_), 2}));
    fft_seq_dataset_ = file_.createDataSet<size_t>(
        "beam_fft_seq_nums", DataSpace({0, 2}, {DataSpace::UNLIMITED, 2}),
        fft_seq_props);
  }

  void process_block(const FFTBlock<T> &block) override {

    const float *fft_ptr = reinterpret_cast<const float *>(&block.fft_output);
    fft_buffer_.insert(fft_buffer_.end(), fft_ptr, fft_ptr + element_count_);

    seq_buffer_.push_back(block.start_seq_num);
    seq_buffer_.push_back(block.end_seq_num);
    buffer_count_++;

    if (buffer_count_ >= batch_size_) {
      flush();
    }
  }

  void flush() override {

    auto current_size = fft_dataset_.getDimensions()[0];

    std::vector<size_t> new_dims = {current_size + buffer_count_};
    new_dims.insert(new_dims.end(), fft_dims_.begin(), fft_dims_.end());
    fft_dataset_.resize(new_dims);

    std::vector<size_t> fft_offset = {current_size};
    fft_offset.insert(fft_offset.end(), fft_dims_.size(), 0);
    std::vector<size_t> fft_count = {buffer_count_};
    fft_count.insert(fft_count.end(), fft_dims_.begin(), fft_dims_.end());

    fft_dataset_.select(fft_offset, fft_count).write_raw(fft_buffer_.data());

    auto seq_size = fft_seq_dataset_.getDimensions()[0];
    fft_seq_dataset_.resize({seq_size + buffer_count_, 2});
    fft_seq_dataset_.select({seq_size, 0}, {buffer_count_, 2})
        .write_raw(seq_buffer_.data());

    fft_buffer_.clear();
    seq_buffer_.clear();
    buffer_count_ = 0;
  }

private:
  HighFive::File &file_;
  size_t element_count_;
  HighFive::DataSet fft_dataset_;
  HighFive::DataSet fft_seq_dataset_;
  HighFive::DataSet fft_idx_dataset_;
  std::vector<size_t> fft_dims_;

  size_t batch_size_;
  size_t buffer_count_ = 0;
  size_t val_elements_ = 1;
  size_t vec_elements_ = 1;

  std::vector<float> fft_buffer_;
  std::vector<size_t> seq_buffer_;
};

template <typename TVal, typename TVec>
class HDF5EigenWriter : public EigenWriter<TVal, TVec> {
public:
  // deflate_level: 0 = no compression (fast, recommended for real-time writes),
  //               1-9 = zlib level (smaller files, slower).
  explicit HDF5EigenWriter(HighFive::File &file, int deflate_level = 0,
                           const int num_blocks = 100)
      : EigenWriter<TVal, TVec>(num_blocks), file_(file),
        batch_size_(num_blocks) {
    using namespace HighFive;

    val_dims_ = get_array_dims<TVal>();
    vec_dims_ = get_array_dims<TVec>();
    vec_dims_.push_back(2); // real / imag
                            //

    for (auto d : val_dims_)
      val_elements_ *= d;
    for (auto d : vec_dims_)
      vec_elements_ *= d;

    val_buffer_.reserve(batch_size_ * val_elements_);
    vec_buffer_.reserve(batch_size_ * vec_elements_);
    seq_buffer_.reserve(batch_size_ * 2);

    {
      std::vector<size_t> dims = {0};
      std::vector<size_t> max_dims = {DataSpace::UNLIMITED};
      dims.insert(dims.end(), val_dims_.begin(), val_dims_.end());
      max_dims.insert(max_dims.end(), val_dims_.begin(), val_dims_.end());

      std::vector<hsize_t> chunk = {static_cast<hsize_t>(batch_size_)};
      chunk.insert(chunk.end(), val_dims_.begin(), val_dims_.end());

      DataSetCreateProps props;
      props.add(Chunking(chunk));
      props.add(Deflate(deflate_level));

      val_dataset_ = file_.createDataSet<float>(
          "eigenvalues", DataSpace(dims, max_dims), props);

      val_dataset_.createAttribute<std::string>(
          "DIMENSION_LABELS",
          std::string("time,channel,polarization,eigenvalue_index"));
    }

    {
      std::vector<size_t> dims = {0};
      std::vector<size_t> max_dims = {DataSpace::UNLIMITED};
      dims.insert(dims.end(), vec_dims_.begin(), vec_dims_.end());
      max_dims.insert(max_dims.end(), vec_dims_.begin(), vec_dims_.end());

      std::vector<hsize_t> chunk = {static_cast<hsize_t>(batch_size_)};
      chunk.insert(chunk.end(), vec_dims_.begin(), vec_dims_.end());

      DataSetCreateProps props;
      props.add(Chunking(chunk));
      props.add(Deflate(deflate_level));

      vec_dataset_ = file_.createDataSet<float>(
          "eigenvectors", DataSpace(dims, max_dims), props);

      vec_dataset_.createAttribute<std::string>(
          "DIMENSION_LABELS",
          std::string("time,channel,polarization"
                      "eigenvalue_index,component_index,complex"));
    }

    {
      DataSetCreateProps props;
      props.add(
          Chunking(std::vector<hsize_t>{static_cast<hsize_t>(batch_size_), 2}));

      seq_dataset_ = file_.createDataSet<size_t>(
          "eigendata_seq_nums", DataSpace({0, 2}, {DataSpace::UNLIMITED, 2}),
          props);

      seq_dataset_.createAttribute<std::string>(
          "description", std::string("start_seq, end_seq per block"));
    }

    // ---- Store shape metadata as attributes for easy introspection ---
    file_.createAttribute<int>("eigenvalue_channels",
                               static_cast<int>(val_dims_[0]));
    file_.createAttribute<int>("eigenvalue_N",
                               static_cast<int>(val_dims_.back()));
  }

  void process_block(const EigenBlock<TVal, TVec> &block) override {
    const float *val_ptr = reinterpret_cast<const float *>(&block.eigenvalues);
    val_buffer_.insert(val_buffer_.end(), val_ptr, val_ptr + val_elements_);

    const float *vec_ptr = reinterpret_cast<const float *>(block.eigenvectors);
    vec_buffer_.insert(vec_buffer_.end(), vec_ptr, vec_ptr + vec_elements_);

    seq_buffer_.push_back(block.start_seq_num);
    seq_buffer_.push_back(block.end_seq_num);

    buffer_count_++;

    // Write to disk when batch is full
    if (buffer_count_ >= batch_size_) {
      flush();
    }
  }

  void flush() override {
    if (buffer_count_ == 0)
      return;
    // Locally tracked row count avoids getDimensions() HDF5 metadata
    // round-trips on every flush (three calls eliminated per batch).
    const size_t t = n_blocks_written_;
    std::vector<size_t> val_new_dims = {t + buffer_count_};
    val_new_dims.insert(val_new_dims.end(), val_dims_.begin(), val_dims_.end());
    val_dataset_.resize(val_new_dims);

    std::vector<size_t> val_offset = {t};
    val_offset.insert(val_offset.end(), val_dims_.size(), 0);
    std::vector<size_t> val_count = {buffer_count_};
    val_count.insert(val_count.end(), val_dims_.begin(), val_dims_.end());

    val_dataset_.select(val_offset, val_count).write_raw(val_buffer_.data());

    std::vector<size_t> vec_new_dims = {t + buffer_count_};
    vec_new_dims.insert(vec_new_dims.end(), vec_dims_.begin(), vec_dims_.end());
    vec_dataset_.resize(vec_new_dims);

    std::vector<size_t> vec_offset = {t};
    vec_offset.insert(vec_offset.end(), vec_dims_.size(), 0);
    std::vector<size_t> vec_count = {buffer_count_};
    vec_count.insert(vec_count.end(), vec_dims_.begin(), vec_dims_.end());

    vec_dataset_.select(vec_offset, vec_count).write_raw(vec_buffer_.data());

    seq_dataset_.resize({t + buffer_count_, 2});
    seq_dataset_.select({t, 0}, {buffer_count_, 2})
        .write_raw(seq_buffer_.data());

    n_blocks_written_ += buffer_count_;
    val_buffer_.clear();
    vec_buffer_.clear();
    seq_buffer_.clear();
    buffer_count_ = 0;
  }

private:
  HighFive::File &file_;
  HighFive::DataSet val_dataset_;
  HighFive::DataSet vec_dataset_;
  HighFive::DataSet seq_dataset_;
  std::vector<size_t> val_dims_; // inner shape of TVal, e.g. {CH, POL, POL, N}
  std::vector<size_t>
      vec_dims_; // inner shape of TVec + [2], e.g. {CH, POL, POL, N, N, 2}
  //
  size_t batch_size_;
  size_t buffer_count_ = 0;
  size_t n_blocks_written_ = 0;
  size_t val_elements_ = 1;
  size_t vec_elements_ = 1;

  std::vector<float> val_buffer_;
  std::vector<float> vec_buffer_;
  std::vector<size_t> seq_buffer_;
};
