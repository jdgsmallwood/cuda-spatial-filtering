#pragma once

#include <array>
#include "spatial/logging.hpp"
#include <condition_variable>
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

template <typename T, size_t N = 0> constexpr auto get_array_dims() {
  if constexpr (std::is_array_v<T>) {
    constexpr size_t extent = std::extent_v<T>;
    auto inner = get_array_dims<std::remove_extent_t<T>, N + 1>();
    inner.insert(inner.begin(), extent);
    return inner;
  } else {
    return std::vector<size_t>{};
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
    beam_buffer_ = static_cast<BeamT*>(std::malloc(batch_size_ * sizeof(BeamT)));
    arrivals_buffer_ = static_cast<ArrivalsT*>(std::malloc(batch_size_ * sizeof(ArrivalsT))); 
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
    // beam_props.add(Deflate(1)); // Light compression
    beam_dataset_ =
        file_.createDataSet<beam_type>("beam_data", beam_space, beam_props);

    // Create arrivals dataset
    DataSpace arrivals_space(arrivals_dataset_dims, arrivals_dataset_max_dims);
    DataSetCreateProps arrivals_props;
    arrivals_props.add(Chunking(arrivals_chunk));
    // arrivals_props.add(Deflate(1)); // Light compression
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
    std::free(beam_buffer_);
        std::free(arrivals_buffer_);
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        const int start_seq, const int end_seq) override {
    // Explicit memcpy for maximum performance
    std::memcpy(&beam_buffer_[current_batch_count_], beam_data, sizeof(BeamT));
    std::memcpy(&arrivals_buffer_[current_batch_count_], arrivals_data, sizeof(ArrivalsT));
    
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
    if (current_batch_count_ == 0) return;

    using clock = std::chrono::high_resolution_clock;
    auto batch_start = clock::now();
    
    // Write beam data
    auto cpu_start = clock::now();
    auto current_size = beam_dataset_.getDimensions()[0];
    auto cpu_end = clock::now();
    LOG_DEBUG("Batch flush: getDimensions() took {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());

    cpu_start = clock::now();
    std::vector<size_t> new_dims = {current_size + current_batch_count_};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    beam_dataset_.resize(new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: beam resize({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());

    cpu_start = clock::now();
    std::vector<size_t> beam_offset = {current_size};
    beam_offset.insert(beam_offset.end(), beam_dims_.size(), 0);
    std::vector<size_t> beam_count = {current_batch_count_};
    beam_count.insert(beam_count.end(), beam_dims_.begin(), beam_dims_.end());
    beam_dataset_.select(beam_offset, beam_count).write_raw(beam_buffer_);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: beam write({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());

    // Write arrivals data
    cpu_start = clock::now();
    auto arrivals_size = arrivals_dataset_.getDimensions()[0];
    std::vector<size_t> arrivals_new_dims = {arrivals_size + current_batch_count_};
    arrivals_new_dims.insert(arrivals_new_dims.end(), arrivals_dims_.begin(),
                             arrivals_dims_.end());
    arrivals_dataset_.resize(arrivals_new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: arrivals resize({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());

    cpu_start = clock::now();
    std::vector<size_t> arrivals_offset = {arrivals_size};
    arrivals_offset.insert(arrivals_offset.end(), arrivals_dims_.size(), 0);
    std::vector<size_t> arrivals_count = {current_batch_count_};
    arrivals_count.insert(arrivals_count.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());
    arrivals_dataset_.select(arrivals_offset, arrivals_count)
        .write_raw(arrivals_buffer_);
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: arrivals write({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());

    // Write sequence numbers
    cpu_start = clock::now();
    auto seq_size = beam_seq_dataset_.getDimensions()[0];
    beam_seq_dataset_.resize({seq_size + current_batch_count_, 2});
    
    beam_seq_dataset_.select({seq_size, 0}, {current_batch_count_, 2})
        .write_raw(seq_buffer_.data());
    cpu_end = clock::now();
    LOG_DEBUG("Batch flush: seq write({}) took {} us", current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());

    auto batch_end = clock::now();
    LOG_DEBUG("Batch flush complete: {} blocks in {} us (avg {} us/block)", 
              current_batch_count_,
              std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count(),
              std::chrono::duration_cast<std::chrono::microseconds>(batch_end - batch_start).count() / current_batch_count_);

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
  BeamT* beam_buffer_;
  ArrivalsT* arrivals_buffer_;
  std::vector<std::array<int,2>> seq_buffer_;
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
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    std::vector<size_t> new_dims = {current_size + 1};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam new_dims construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    beam_dataset_.resize(new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_dataset_.resize(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    std::vector<size_t> beam_offset = {current_size};
    beam_offset.insert(beam_offset.end(), beam_dims_.size(), 0);
    std::vector<size_t> beam_count = {1};
    beam_count.insert(beam_count.end(), beam_dims_.begin(), beam_dims_.end());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam offset/count construction: {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    using beam_type = std::remove_all_extents<BeamT>::type;
    beam_dataset_.select(beam_offset, beam_count).write_raw(beam_data);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_dataset_.select().write(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    // Write arrivals data
    cpu_start = clock::now();
    auto arrivals_size = arrivals_dataset_.getDimensions()[0];
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals_dataset_.getDimensions(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
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
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    arrivals_dataset_.resize(arrivals_new_dims);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals_dataset_.resize(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    arrivals_dataset_.select(arrivals_offset, arrivals_count).write_raw(arrivals_data);
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for arrivals_dataset_.select().write_raw(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    // Write sequence numbers
    cpu_start = clock::now();
    auto seq_size = beam_seq_dataset_.getDimensions()[0];
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_seq_dataset_.getDimensions(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    beam_seq_dataset_.resize({seq_size + 1, 2});
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_seq_dataset_.resize(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
    
    cpu_start = clock::now();
    std::vector<int> seq_nums = {start_seq, end_seq};
    beam_seq_dataset_.select({seq_size, 0}, {1, 2}).write_raw(seq_nums.data());
    cpu_end = clock::now();
    LOG_DEBUG("CPU overhead for beam_seq_dataset_.select().write_raw(): {} us",
              std::chrono::duration_cast<std::chrono::microseconds>(cpu_end - cpu_start).count());
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
