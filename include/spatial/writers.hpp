#pragma once

#include "spatial/logging.hpp"
#include <array>
#include <condition_variable>
#include <fstream>
#include <hdf5.h>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

template <typename BeamT, typename ArrivalsT> class BeamWriter {
public:
  virtual ~BeamWriter() = default;
  virtual void write_beam_block(const BeamT *beam_data,
                                const ArrivalsT *arrivals_data,
                                const int start_seq, const int end_seq) = 0;
  virtual void flush() = 0;
};

template <typename T> class VisibilitiesWriter {
public:
  virtual ~VisibilitiesWriter() = default;
  virtual void write_visibilities_block(const T *data, const int start_seq,
                                        const int end_seq) = 0;
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
class BatchedHDF5BeamWriter : public BeamWriter<BeamT, ArrivalsT> {
public:
  BatchedHDF5BeamWriter(HighFive::File &file, size_t batch_size = 100)
      : file_(file), batch_size_(batch_size), current_batch_count_(0) {
    using namespace HighFive;
    using beam_type = std::remove_all_extents<BeamT>::type;
    using arrival_type = std::remove_all_extents<ArrivalsT>::type;

    beam_element_count_ = sizeof(BeamT) / sizeof(beam_type);
    arrivals_element_count_ = sizeof(ArrivalsT) / sizeof(bool);
    beam_dims_ = get_array_dims<BeamT>();
    arrivals_dims_ = get_array_dims<ArrivalsT>();

    // Pre-allocate fixed-size buffers
    beam_buffer_ =
        static_cast<BeamT *>(std::malloc(batch_size_ * sizeof(BeamT)));
    arrivals_buffer_ =
        static_cast<ArrivalsT *>(std::malloc(batch_size_ * sizeof(ArrivalsT)));
    seq_buffer_.resize(batch_size_);

    // Setup datasets
    std::vector<size_t> beam_dataset_dims = {0};
    std::vector<size_t> beam_dataset_max_dims = {DataSpace::UNLIMITED};
    beam_dataset_dims.insert(beam_dataset_dims.end(), beam_dims_.begin(),
                             beam_dims_.end());
    beam_dataset_max_dims.insert(beam_dataset_max_dims.end(),
                                 beam_dims_.begin(), beam_dims_.end());

    // Chunk size matches batch size for optimal I/O
    std::vector<hsize_t> beam_chunk = {batch_size_};
    beam_chunk.insert(beam_chunk.end(), beam_dims_.begin(), beam_dims_.end());

    std::vector<size_t> arrivals_dataset_dims = {0};
    std::vector<size_t> arrivals_dataset_max_dims = {DataSpace::UNLIMITED};
    arrivals_dataset_dims.insert(arrivals_dataset_dims.end(),
                                 arrivals_dims_.begin(), arrivals_dims_.end());
    arrivals_dataset_max_dims.insert(arrivals_dataset_max_dims.end(),
                                     arrivals_dims_.begin(),
                                     arrivals_dims_.end());

    std::vector<hsize_t> arrivals_chunk = {batch_size_};
    arrivals_chunk.insert(arrivals_chunk.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());

    // Create beam dataset
    DataSpace beam_space(beam_dataset_dims, beam_dataset_max_dims);
    DataSetCreateProps beam_props;
    beam_props.add(Chunking(beam_chunk));
    // Optional: reduce compression for speed
    beam_props.add(Deflate(4)); // Light compression
    beam_dataset_ =
        file_.createDataSet<beam_type>("beam_data", beam_space, beam_props);

    // Create arrivals dataset
    DataSpace arrivals_space(arrivals_dataset_dims, arrivals_dataset_max_dims);
    DataSetCreateProps arrivals_props;
    arrivals_props.add(Chunking(arrivals_chunk));
    arrivals_props.add(Deflate(4)); // Light compression
    arrivals_dataset_ = file_.createDataSet<arrival_type>(
        "arrivals", arrivals_space, arrivals_props);

    // Create sequence number dataset
    DataSetCreateProps beam_seq_props;
    beam_seq_props.add(Chunking(std::vector<hsize_t>{batch_size_, 2}));
    beam_seq_dataset_ = file_.createDataSet<int>(
        "beam_seq_nums", DataSpace({0, 2}, {DataSpace::UNLIMITED, 2}),
        beam_seq_props);
  }

  ~BatchedHDF5BeamWriter() {
    // Ensure any remaining buffered data is written
    flush_batch();
    free(beam_buffer_);
    free(arrivals_buffer_);
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        const int start_seq, const int end_seq) override {
    // Explicit memcpy for maximum performance
    std::memcpy(&beam_buffer_[current_batch_count_], beam_data, sizeof(BeamT));
    std::memcpy(&arrivals_buffer_[current_batch_count_], arrivals_data,
                sizeof(ArrivalsT));

    seq_buffer_[current_batch_count_][0] = start_seq;
    seq_buffer_[current_batch_count_][1] = end_seq;

    current_batch_count_++;

    // Flush when batch is full
    if (current_batch_count_ >= batch_size_) {
      flush_batch();
    }
  }

  void flush() override {
    flush_batch();
    file_.flush();
  }

private:
  void flush_batch() {
    if (current_batch_count_ == 0)
      return;

    using clock = std::chrono::high_resolution_clock;
    auto batch_start = clock::now();

    // Write beam data
    auto cpu_start = clock::now();
    auto current_size = beam_dataset_.getDimensions()[0];
    auto cpu_end = clock::now();
    LOG_DEBUG("Batch flush: getDimensions() took {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<size_t> new_dims = {current_size + current_batch_count_};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    beam_dataset_.resize(new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: beam resize({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<size_t> beam_offset = {current_size};
    beam_offset.insert(beam_offset.end(), beam_dims_.size(), 0);
    std::vector<size_t> beam_count = {current_batch_count_};
    beam_count.insert(beam_count.end(), beam_dims_.begin(), beam_dims_.end());
    beam_dataset_.select(beam_offset, beam_count).write_raw(beam_buffer_);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: beam write({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Write arrivals data
    cpu_start = clock::now();
    auto arrivals_size = arrivals_dataset_.getDimensions()[0];
    std::vector<size_t> arrivals_new_dims = {arrivals_size +
                                             current_batch_count_};
    arrivals_new_dims.insert(arrivals_new_dims.end(), arrivals_dims_.begin(),
                             arrivals_dims_.end());
    arrivals_dataset_.resize(arrivals_new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: arrivals resize({}) took {} us",
              current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<size_t> arrivals_offset = {arrivals_size};
    arrivals_offset.insert(arrivals_offset.end(), arrivals_dims_.size(), 0);
    std::vector<size_t> arrivals_count = {current_batch_count_};
    arrivals_count.insert(arrivals_count.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());
    arrivals_dataset_.select(arrivals_offset, arrivals_count)
        .write_raw(arrivals_buffer_);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: arrivals write({}) took {} us",
              current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Write sequence numbers
    cpu_start = clock::now();
    auto seq_size = beam_seq_dataset_.getDimensions()[0];
    beam_seq_dataset_.resize({seq_size + current_batch_count_, 2});

    beam_seq_dataset_.select({seq_size, 0}, {current_batch_count_, 2})
        .write_raw(seq_buffer_.data());
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: seq write({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    auto batch_end = clock::now();
    LOG_DEBUG("Batch flush complete: {} blocks in {} us (avg {} us/block)",
              current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(batch_end -
                                                                    batch_start)
                  .count(),
              std::chrono::duration_cast<std::chrono::microseconds>(batch_end -
                                                                    batch_start)
                      .count() /
                  current_batch_count_);

    // Reset counter (no need to clear vectors - we'll overwrite)
    current_batch_count_ = 0;
  }

  HighFive::File &file_;
  size_t batch_size_;
  size_t current_batch_count_;
  size_t beam_element_count_;
  size_t arrivals_element_count_;
  std::vector<size_t> beam_dims_;
  std::vector<size_t> arrivals_dims_;
  HighFive::DataSet beam_dataset_;
  HighFive::DataSet arrivals_dataset_;
  HighFive::DataSet beam_seq_dataset_;

  // Batching buffers - store complete structures
  BeamT *beam_buffer_;
  ArrivalsT *arrivals_buffer_;
  std::vector<std::array<int, 2>> seq_buffer_;
};

template <typename BeamT, typename ArrivalsT>
class HDF5BeamWriter : public BeamWriter<BeamT, ArrivalsT> {

public:
  HDF5BeamWriter(HighFive::File &file) : file_(file) {
    using namespace HighFive;
    using beam_type = std::remove_all_extents<BeamT>::type;
    using arrival_type = std::remove_all_extents<ArrivalsT>::type;
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
    beam_dataset_ =
        file_.createDataSet<beam_type>("beam_data", beam_space, beam_props);

    // Create arrivals dataset
    DataSpace arrivals_space(arrivals_dataset_dims, arrivals_dataset_max_dims);
    DataSetCreateProps arrivals_props;
    arrivals_props.add(Chunking(arrivals_chunk));
    arrivals_dataset_ = file_.createDataSet<arrival_type>(
        "arrivals", arrivals_space, arrivals_props);

    DataSetCreateProps beam_seq_props;
    beam_seq_props.add(Chunking(std::vector<hsize_t>{1, 2}));
    // Create sequence number dataset
    beam_seq_dataset_ = file_.createDataSet<int>(
        "beam_seq_nums", DataSpace({0, 2}, {DataSpace::UNLIMITED, 2}),
        beam_seq_props);
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        const int start_seq, const int end_seq) override {
    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();

    // Write beam data
    cpu_start = clock::now();
    auto current_size = beam_dataset_.getDimensions()[0];
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_dataset_.getDimensions(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<size_t> new_dims = {current_size + 1};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam new_dims construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    beam_dataset_.resize(new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_dataset_.resize(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<size_t> beam_offset = {current_size};
    beam_offset.insert(beam_offset.end(), beam_dims_.size(), 0);
    std::vector<size_t> beam_count = {1};
    beam_count.insert(beam_count.end(), beam_dims_.begin(), beam_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam offset/count construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    using beam_type = std::remove_all_extents<BeamT>::type;
    beam_dataset_.select(beam_offset, beam_count).write_raw(beam_data);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_dataset_.select().write(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Write arrivals data
    cpu_start = clock::now();
    auto arrivals_size = arrivals_dataset_.getDimensions()[0];
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals_dataset_.getDimensions(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<size_t> arrivals_new_dims = {arrivals_size + 1};
    arrivals_new_dims.insert(arrivals_new_dims.end(), arrivals_dims_.begin(),
                             arrivals_dims_.end());
    std::vector<size_t> arrivals_offset = {arrivals_size};
    arrivals_offset.insert(arrivals_offset.end(), arrivals_dims_.size(), 0);
    std::vector<size_t> arrivals_count = {1};
    arrivals_count.insert(arrivals_count.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals dims/offset/count construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    arrivals_dataset_.resize(arrivals_new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals_dataset_.resize(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    arrivals_dataset_.select(arrivals_offset, arrivals_count)
        .write_raw(arrivals_data);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals_dataset_.select().write_raw(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Write sequence numbers
    cpu_start = clock::now();
    auto seq_size = beam_seq_dataset_.getDimensions()[0];
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_seq_dataset_.getDimensions(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    beam_seq_dataset_.resize({seq_size + 1, 2});
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_seq_dataset_.resize(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<int> seq_nums = {start_seq, end_seq};
    beam_seq_dataset_.select({seq_size, 0}, {1, 2}).write_raw(seq_nums.data());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_seq_dataset_.select().write_raw(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());
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
};

template <typename T>
class HDF5VisibilitiesWriter : public VisibilitiesWriter<T> {
public:
  HDF5VisibilitiesWriter(HighFive::File &file)
      : file_(file),
        element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)) {
    using namespace HighFive;
    vis_dims_ = get_array_dims<T>();
    std::vector<size_t> vis_dataset_dims = {0};
    std::vector<size_t> vis_dataset_max_dims = {DataSpace::UNLIMITED};

    vis_dataset_dims.insert(vis_dataset_dims.end(), vis_dims_.begin(),
                            vis_dims_.end());
    vis_dataset_max_dims.insert(vis_dataset_max_dims.end(), vis_dims_.begin(),
                                vis_dims_.end());
    std::vector<hsize_t> vis_chunk = {1};
    vis_chunk.insert(vis_chunk.end(), vis_dims_.begin(), vis_dims_.end());
    DataSpace vis_space(vis_dataset_dims, vis_dataset_max_dims);
    DataSetCreateProps props;
    props.add(HighFive::Chunking(vis_chunk));
    // I imagine createDataSet needs a primitive type
    vis_dataset_ = file_.createDataSet<float>("visibilities", vis_space, props);

    DataSetCreateProps vis_seq_props;
    vis_seq_props.add(Chunking(std::vector<hsize_t>{1, 2}));
    vis_seq_dataset_ = file_.createDataSet<int>(
        "vis_seq_nums", DataSpace({0, 2}, {HighFive::DataSpace::UNLIMITED, 2}),
        vis_seq_props);
  }

  void write_visibilities_block(const T *data, const int start_seq,
                                const int end_seq) override {
    LOG_INFO("writing visibilities block {} to {}", start_seq, end_seq);
    auto current_size = vis_dataset_.getDimensions()[0];
    std::vector<size_t> new_dims = {current_size + 1};
    new_dims.insert(new_dims.end(), vis_dims_.begin(), vis_dims_.end());
    vis_dataset_.resize(new_dims);

    std::vector<size_t> vis_offset = {current_size};
    vis_offset.insert(vis_offset.end(), vis_dims_.size(), 0);
    std::vector<size_t> vis_count = {1};
    vis_count.insert(vis_count.end(), vis_dims_.begin(), vis_dims_.end());

    vis_dataset_.select(vis_offset, vis_count).write(data);

    auto seq_size = vis_seq_dataset_.getDimensions()[0];
    vis_seq_dataset_.resize({seq_size + 1, 2});
    std::vector<int> seq_nums = {start_seq, end_seq};
    vis_seq_dataset_.select({seq_size, 0}, {1, 2}).write_raw(seq_nums.data());
  }

  void flush() override { file_.flush(); }

private:
  HighFive::File &file_;
  size_t element_count_;
  HighFive::DataSet vis_dataset_;
  HighFive::DataSet vis_seq_dataset_;
  std::vector<size_t> vis_dims_;
};

// Factory function for easy creation
// template <typename T>
// std::unique_ptr<BufferedOutput<T>>
// create_hdf5_output(const std::string &filename, size_t beam_buffer_size,
//                   size_t vis_buffer_size) {
//  auto file =
//      std::make_shared<HighFive::File>(filename, HighFive::File::Truncate);
//
//  auto beam_writer =
//      std::make_unique<HDF5BeamWriter<typename T::BeamOutputType,
//                                      typename T::ArrivalOutputType>>(*file);
//  auto vis_writer = std::make_unique<
//      HDF5VisibilitiesWriter<typename T::VisibilitiesOutputType>>(*file);
//
//  return std::make_unique<BufferedOutput<T>>(std::move(beam_writer),
//                                             std::move(vis_writer),
//                                             beam_buffer_size,
//                                             vis_buffer_size);
//}
//

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
class HDF5RawBeamWriter : public BeamWriter<BeamT, ArrivalsT> {
public:
  HDF5RawBeamWriter(hid_t file_id) : file_id_(file_id) {
    using beam_type = std::remove_all_extents_t<BeamT>;
    using arrival_type = std::remove_all_extents_t<ArrivalsT>;

    beam_element_count_ = sizeof(BeamT) / sizeof(beam_type);
    arrivals_element_count_ = sizeof(ArrivalsT) / sizeof(arrival_type);
    beam_dims_ = get_array_dims<BeamT, hsize_t>();
    arrivals_dims_ = get_array_dims<ArrivalsT, hsize_t>();

    // Create beam dataset
    beam_dataset_id_ = create_dataset<beam_type>("beam_data", beam_dims_);

    // Create arrivals dataset
    arrivals_dataset_id_ =
        create_dataset<arrival_type>("arrivals", arrivals_dims_);

    // Create sequence number dataset (2D: Nx2)
    std::vector<hsize_t> seq_dims = {2}; // columns: start_seq, end_seq
    beam_seq_dataset_id_ = create_dataset<int>("beam_seq_nums", seq_dims);
  }

  ~HDF5RawBeamWriter() {
    if (beam_dataset_id_ >= 0)
      H5Dclose(beam_dataset_id_);
    if (arrivals_dataset_id_ >= 0)
      H5Dclose(arrivals_dataset_id_);
    if (beam_seq_dataset_id_ >= 0)
      H5Dclose(beam_seq_dataset_id_);
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        const int start_seq, const int end_seq) override {
    using clock = std::chrono::high_resolution_clock;
    auto cpu_start = clock::now();
    auto cpu_end = clock::now();

    // Write beam data
    cpu_start = clock::now();
    hsize_t current_size = get_dataset_size(beam_dataset_id_);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam dataset size query: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<hsize_t> new_dims = {current_size + 1};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam new_dims construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    resize_dataset(beam_dataset_id_, new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam dataset resize: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<hsize_t> offset = {current_size};
    offset.insert(offset.end(), beam_dims_.size(), 0);
    std::vector<hsize_t> count = {1};
    count.insert(count.end(), beam_dims_.begin(), beam_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam offset/count construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    using beam_type = std::remove_all_extents_t<BeamT>;
    write_hyperslab<beam_type>(beam_dataset_id_, (beam_type *)beam_data, offset,
                               count, beam_dims_);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam dataset write: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Write arrivals data
    cpu_start = clock::now();
    hsize_t arrivals_size = get_dataset_size(arrivals_dataset_id_);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals dataset size query: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    std::vector<hsize_t> arrivals_new_dims = {arrivals_size + 1};
    arrivals_new_dims.insert(arrivals_new_dims.end(), arrivals_dims_.begin(),
                             arrivals_dims_.end());
    std::vector<hsize_t> arrivals_offset = {arrivals_size};
    arrivals_offset.insert(arrivals_offset.end(), arrivals_dims_.size(), 0);
    std::vector<hsize_t> arrivals_count = {1};
    arrivals_count.insert(arrivals_count.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals dims/offset/count construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    resize_dataset(arrivals_dataset_id_, arrivals_new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals dataset resize: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    using arrival_type = std::remove_all_extents_t<ArrivalsT>;
    write_hyperslab<arrival_type>(
        arrivals_dataset_id_, (arrival_type *)arrivals_data, arrivals_offset,
        arrivals_count, arrivals_dims_);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals dataset write: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    // Write sequence numbers
    cpu_start = clock::now();
    hsize_t seq_size = get_dataset_size(beam_seq_dataset_id_);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for seq dataset size query: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    resize_dataset(beam_seq_dataset_id_, {seq_size + 1, 2});
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for seq dataset resize: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());

    cpu_start = clock::now();
    int seq_nums[2] = {start_seq, end_seq};
    write_hyperslab<int>(beam_seq_dataset_id_, seq_nums, {seq_size, 0}, {1, 2},
                         {2});
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for seq dataset write: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end -
                                                                    cpu_start)
                  .count());
  }

  void flush() override { H5Fflush(file_id_, H5F_SCOPE_GLOBAL); }

private:
  template <typename T>
  hid_t create_dataset(const char *name,
                       const std::vector<hsize_t> &data_dims) {
    // Initial dimensions: 0 in first dimension, rest from data_dims
    std::vector<hsize_t> dims = {0};
    dims.insert(dims.end(), data_dims.begin(), data_dims.end());

    // Max dimensions: unlimited in first, rest from data_dims
    std::vector<hsize_t> max_dims = {H5S_UNLIMITED};
    max_dims.insert(max_dims.end(), data_dims.begin(), data_dims.end());

    // Chunk dimensions: 1 in first, rest from data_dims
    std::vector<hsize_t> chunk_dims = {1};
    chunk_dims.insert(chunk_dims.end(), data_dims.begin(), data_dims.end());

    // Create dataspace
    hid_t space_id =
        H5Screate_simple(dims.size(), dims.data(), max_dims.data());

    // Create chunked dataset creation property list
    hid_t plist_id = H5Pcreate(H5P_DATASET_CREATE);
    H5Pset_chunk(plist_id, chunk_dims.size(), chunk_dims.data());

    // Create dataset
    hid_t dataset_id = H5Dcreate2(file_id_, name, get_hdf5_type<T>(), space_id,
                                  H5P_DEFAULT, plist_id, H5P_DEFAULT);

    H5Pclose(plist_id);
    H5Sclose(space_id);

    return dataset_id;
  }

  hsize_t get_dataset_size(hid_t dataset_id) {
    hid_t space_id = H5Dget_space(dataset_id);
    int ndims = H5Sget_simple_extent_ndims(space_id);
    std::vector<hsize_t> dims(ndims);
    H5Sget_simple_extent_dims(space_id, dims.data(), nullptr);
    H5Sclose(space_id);
    return dims[0];
  }

  void resize_dataset(hid_t dataset_id, const std::vector<hsize_t> &new_dims) {
    H5Dset_extent(dataset_id, new_dims.data());
  }

  template <typename T>
  void write_hyperslab(hid_t dataset_id, const T *data,
                       const std::vector<hsize_t> &offset,
                       const std::vector<hsize_t> &count,
                       const std::vector<hsize_t> &data_dims) {
    // Get file dataspace and select hyperslab
    hid_t file_space_id = H5Dget_space(dataset_id);
    H5Sselect_hyperslab(file_space_id, H5S_SELECT_SET, offset.data(), nullptr,
                        count.data(), nullptr);

    // Create memory dataspace
    hid_t mem_space_id = H5Screate_simple(count.size(), count.data(), nullptr);

    // Write data
    H5Dwrite(dataset_id, get_hdf5_type<T>(), mem_space_id, file_space_id,
             H5P_DEFAULT, data);

    H5Sclose(mem_space_id);
    H5Sclose(file_space_id);
  }

  hid_t file_id_;
  hid_t beam_dataset_id_;
  hid_t arrivals_dataset_id_;
  hid_t beam_seq_dataset_id_;

  size_t beam_element_count_;
  std::vector<hsize_t> beam_dims_;
  size_t arrivals_element_count_;
  std::vector<hsize_t> arrivals_dims_;
};

template <typename BeamT, typename ArrivalsT>
class BinaryRawBeamWriter : public BeamWriter<BeamT, ArrivalsT> {
public:
  BinaryRawBeamWriter(const std::string &filename)
      : filename_(filename),
        file_stream_(filename, std::ios::binary | std::ios::out) {
    if (!file_stream_) {
      throw std::runtime_error("Failed to open binary file: " + filename);
    }

    using beam_type = std::remove_all_extents_t<BeamT>;
    using arrival_type = std::remove_all_extents_t<ArrivalsT>;

    beam_element_count_ = sizeof(BeamT) / sizeof(beam_type);
    arrivals_element_count_ = sizeof(ArrivalsT) / sizeof(arrival_type);

    std::cout << "[BinaryRawBeamWriter] Opened " << filename
              << " for writing.\n";
  }

  ~BinaryRawBeamWriter() {
    if (file_stream_.is_open()) {
      file_stream_.close();
      std::cout << "[BinaryRawBeamWriter] Closed " << filename_ << ".\n";
    }
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        const int start_seq, const int end_seq) {
    using clock = std::chrono::high_resolution_clock;

    auto cpu_start = clock::now();

    // Write header: start_seq, end_seq, element counts
    file_stream_.write(reinterpret_cast<const char *>(&start_seq), sizeof(int));
    file_stream_.write(reinterpret_cast<const char *>(&end_seq), sizeof(int));

    // Write beam data
    file_stream_.write(reinterpret_cast<const char *>(beam_data),
                       sizeof(BeamT));

    // Write arrivals data
    file_stream_.write(reinterpret_cast<const char *>(arrivals_data),
                       sizeof(ArrivalsT));

    auto cpu_end = clock::now();

    auto elapsed_us = std::chrono::duration_cast<std::chrono::microseconds>(
                          cpu_end - cpu_start)
                          .count();

    std::cout << "[BinaryRawBeamWriter] Wrote beam block ("
              << sizeof(BeamT) + sizeof(ArrivalsT) << " bytes) "
              << "in " << elapsed_us << " us.\n";
  }

  void flush() {
    file_stream_.flush();
    std::cout << "[BinaryRawBeamWriter] Flushed file buffer.\n";
  }

private:
  std::string filename_;
  std::ofstream file_stream_;

  size_t beam_element_count_;
  size_t arrivals_element_count_;
};

template <typename BeamT, typename ArrivalsT>
class InMemoryBeamWriter : public BeamWriter<BeamT, ArrivalsT> {
public:
  struct Meta {
    int start_seq;
    int end_seq;
  };

  explicit InMemoryBeamWriter(size_t capacity)
      : capacity_(capacity), start_index_(0), count_(0) {
    beam_buffer_ = static_cast<BeamT *>(std::malloc(capacity * sizeof(BeamT)));
    arrivals_buffer_ =
        static_cast<ArrivalsT *>(std::malloc(capacity * sizeof(ArrivalsT)));
    meta_buffer_ = static_cast<Meta *>(std::malloc(capacity * sizeof(Meta)));

    if (!beam_buffer_ || !arrivals_buffer_ || !meta_buffer_) {
      throw std::bad_alloc();
    }

    std::cout << "[InMemoryRawBeamWriter] Allocated space for " << capacity
              << " blocks (" << sizeof(BeamT) << " + " << sizeof(ArrivalsT)
              << " bytes each)\n";
  }

  ~InMemoryBeamWriter() override {
    std::free(beam_buffer_);
    std::free(arrivals_buffer_);
    std::free(meta_buffer_);
    std::cout << "[InMemoryRawBeamWriter] Freed buffers.\n";
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        int start_seq, int end_seq) override {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();

    size_t write_index = (start_index_ + count_) % capacity_;
    if (count_ == capacity_) {
      // Overwrite oldest
      start_index_ = (start_index_ + 1) % capacity_;
      std::cout
          << "[InMemoryRawBeamWriter] Buffer full, overwriting oldest block.\n";
    } else {
      ++count_;
    }

    meta_buffer_[write_index].start_seq = start_seq;
    meta_buffer_[write_index].end_seq = end_seq;

    std::memcpy(&beam_buffer_[write_index], beam_data, sizeof(BeamT));
    std::memcpy(&arrivals_buffer_[write_index], arrivals_data,
                sizeof(ArrivalsT));

    auto end = clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start)
                  .count();

    std::cout << "[InMemoryRawBeamWriter] Copied block " << write_index
              << " in " << us << " µs.\n";
  }

  void flush() override {
    std::cout << "[InMemoryRawBeamWriter] Flush (no-op).\n";
  }

  size_t size() const { return count_; }
  size_t capacity() const { return capacity_; }

  void clear() {
    start_index_ = 0;
    count_ = 0;
  }

  // Retrieve a copy of a block by index (0 = oldest)
  void get_block(size_t i, int &start_seq, int &end_seq, BeamT &beam,
                 ArrivalsT &arrivals) const {
    if (i >= count_)
      throw std::out_of_range("get_block index out of range");

    size_t idx = (start_index_ + i) % capacity_;
    start_seq = meta_buffer_[idx].start_seq;
    end_seq = meta_buffer_[idx].end_seq;

    std::memcpy(&beam, &beam_buffer_[idx], sizeof(BeamT));
    std::memcpy(&arrivals, &arrivals_buffer_[idx], sizeof(ArrivalsT));
  }

private:
  size_t capacity_;
  size_t start_index_;
  size_t count_;

  BeamT *beam_buffer_;
  ArrivalsT *arrivals_buffer_;
  Meta *meta_buffer_;
};
