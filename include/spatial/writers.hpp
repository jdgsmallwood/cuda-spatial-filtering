#pragma once

#include "spatial/logging.hpp"
#include <array>
#include <condition_variable>
#include <fitsio.h>
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
                                        const int end_seq,
                                        const int num_missing_packets,
                                        const int num_total_packets) = 0;
  virtual void flush() = 0;
};

template <typename TVal, typename TVec> class EigenWriter {
public:
  virtual ~EigenWriter() = default;
  virtual void write_eigendata_block(const TVal *val_data, const TVec *vec_data,
                                     const int start_seq,
                                     const int end_seq) = 0;
  virtual void flush() = 0;
};

template <typename T> class FFTWriter {
public:
  virtual ~FFTWriter() = default;
  virtual void write_fft_block(const T *fft_data, const int start_seq,
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
    using beam_type = typename std::remove_all_extents<BeamT>::type;
    using arrival_type = typename std::remove_all_extents<ArrivalsT>::type;

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
    using beam_type = typename std::remove_all_extents<BeamT>::type;
    using arrival_type = typename std::remove_all_extents<ArrivalsT>::type;
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
    using beam_type = typename std::remove_all_extents<BeamT>::type;
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
  HDF5VisibilitiesWriter(
      HighFive::File &file, const int min_channel, const int max_channel,
      const std::unordered_map<int, int> *antenna_map = nullptr)
      : file_(file),
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

    DataSetCreateProps vis_missing_props;
    vis_missing_props.add(Chunking(std::vector<hsize_t>{1, 3}));
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

  void write_visibilities_block(const T *data, const int start_seq,
                                const int end_seq,
                                const int num_missing_packets,
                                const int num_total_packets) override {
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

    auto missing_size = vis_missing_dataset_.getDimensions()[0];
    vis_missing_dataset_.resize({missing_size + 1, 3});
    float num_missing_packets_fl = static_cast<float>(num_missing_packets);
    float num_total_packets_fl = static_cast<float>(num_total_packets);
    std::vector<float> missing_nums = {
        num_missing_packets_fl, num_total_packets_fl,
        100 * num_missing_packets_fl / num_total_packets_fl};
    vis_missing_dataset_.select({missing_size, 0}, {1, 3})
        .write_raw(missing_nums.data());
  }

  void flush() override { file_.flush(); }

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
};

template <typename T>
class HDF5AndRedisVisibilitiesWriter : public VisibilitiesWriter<T> {
public:
  HDF5AndRedisVisibilitiesWriter(
      HighFive::File &file, const int NR_BASELINES, const int min_channel,
      const int max_channel,
      const std::unordered_map<int, int> *antenna_map = nullptr)
      : file_(file),
        element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)),
        redis("tcp://127.0.0.1:6379"), NR_BASELINES(NR_BASELINES) {
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

    DataSetCreateProps vis_missing_props;
    vis_missing_props.add(Chunking(std::vector<hsize_t>{1, 3}));
    vis_missing_dataset_ = file_.createDataSet<float>(
        "vis_missing_nums",
        DataSpace({0, 3}, {HighFive::DataSpace::UNLIMITED, 3}),
        vis_missing_props);

    if (antenna_map && !antenna_map->empty()) {
      this->antenna_map_ = *antenna_map;
    } else {
      generate_identity_map();
    }

    NR_CHANNELS = vis_dims_[0];
    NR_POLARIZATIONS = vis_dims_[2];
    write_baseline_ids();
    create_all_timeseries_keys();
  }

  void write_visibilities_block(const T *data, const int start_seq,
                                const int end_seq,
                                const int num_missing_packets,
                                const int num_total_packets) override {
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

    auto missing_size = vis_missing_dataset_.getDimensions()[0];
    vis_missing_dataset_.resize({missing_size + 1, 3});
    float num_missing_packets_fl = static_cast<float>(num_missing_packets);
    float num_total_packets_fl = static_cast<float>(num_total_packets);
    std::vector<float> missing_nums = {
        num_missing_packets_fl, num_total_packets_fl,
        100 * num_missing_packets_fl / num_total_packets_fl};
    vis_missing_dataset_.select({missing_size, 0}, {1, 3})
        .write_raw(missing_nums.data());
    auto ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::system_clock::now().time_since_epoch())
                  .count();

    std::vector<std::string> madd_args = {"TS.MADD"};

    // Reserve space for efficiency: 4 metrics * 3 arguments (Key, TS, Value)
    madd_args.reserve(1 + NR_CHANNELS * NR_BASELINES * 1 * 1 * 4 * 3);

    // Iterate over all dimensions
    for (int ch_idx = 0; ch_idx < NR_CHANNELS; ++ch_idx) {
      for (int bl_idx = 0; bl_idx < NR_BASELINES; ++bl_idx) {
        for (int pol_r_idx = 0; pol_r_idx < 1; ++pol_r_idx) {
          for (int pol_c_idx = 0; pol_c_idx < 1; ++pol_c_idx) {

            // --- 1. Data Access & Calculation ---
            const float real_val =
                data[0][ch_idx][bl_idx][pol_r_idx][pol_c_idx][0];
            const float imag_val =
                data[0][ch_idx][bl_idx][pol_r_idx][pol_c_idx][1];

            const float amplitude =
                std::sqrt(real_val * real_val + imag_val * imag_val);
            const float phase = std::atan2(imag_val, real_val);

            // --- 2. Create Descriptive Strings ---
            std::string channel_id = std::to_string(ch_idx);
            std::string baseline_pair = std::to_string(bl_idx);
            std::string pol_pair =
                std::to_string(pol_r_idx) + "-" + std::to_string(pol_c_idx);

            // --- 3. Key Prefix: ts:ch:<ID>:bl:<A-B>:p:<R-C> ---
            std::string key_prefix = "ts:ch:" + channel_id +
                                     ":bl:" + baseline_pair + ":p:" + pol_pair;

            // Convert timestamp to string once
            std::string ts_str = std::to_string(ts);

            // --- 4. Build TS.MADD Arguments (4 metrics per data point) ---

            // Metric 1: Real
            // madd_args.push_back(key_prefix + ":real");
            // madd_args.push_back(ts_str);
            // madd_args.push_back(std::to_string(real_val));

            //// Metric 2: Imaginary
            // madd_args.push_back(key_prefix + ":imag");
            // madd_args.push_back(ts_str);
            // madd_args.push_back(std::to_string(imag_val));

            // Metric 3: Amplitude
            madd_args.push_back(key_prefix + ":amp");
            madd_args.push_back(ts_str);
            madd_args.push_back(std::to_string(amplitude));

            // Metric 4: Phase
            madd_args.push_back(key_prefix + ":phase");
            madd_args.push_back(ts_str);
            madd_args.push_back(std::to_string(phase));
          }
        }
      }
    }

    // --- 5. Execute TS.MADD using redis-plus-plus ---
    if (madd_args.size() > 1) { // Check if we have arguments beyond "TS.MADD"
      // This is the correct call signature for redis-plus-plus command with a
      // list of arguments The return type can often be void or a future/reply
      // type depending on sync/async usage
      redis.command(madd_args.begin(), madd_args.end());
    }
  }

  void flush() override { file_.flush(); }

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
  void create_all_timeseries_keys() {

    std::cout << "Starting TimeSeries key pre-creation..." << std::endl;
    std::cout << "Total keys to create: "
              << NR_CHANNELS * NR_BASELINES * 4 * NR_POLARIZATIONS *
                     NR_POLARIZATIONS
              << std::endl;

    //    const std::vector<std::string> components = {"real", "imag", "amp",
    //                                                 "phase"};

    const std::vector<std::string> components = {"amp", "phase"};
    for (int ch_idx = 0; ch_idx < NR_CHANNELS; ++ch_idx) {
      std::string channel_id = std::to_string(ch_idx);

      for (int bl_idx = 0; bl_idx < NR_BASELINES; ++bl_idx) {
        int ant1, ant2;
        std::string baseline_pair = std::to_string(bl_idx);

        for (int pol_r_idx = 0; pol_r_idx < 1; ++pol_r_idx) {
          for (int pol_c_idx = 0; pol_c_idx < 1; ++pol_c_idx) {
            std::string pol_pair =
                std::to_string(pol_r_idx) + "-" + std::to_string(pol_c_idx);

            for (const auto &component : components) {
              // --- 1. Construct the Key ---
              std::string key = "ts:ch:" + channel_id + ":bl:" + baseline_pair +
                                ":p:" + pol_pair + ":" + component;

              // --- 2. Build the TS.CREATE Command Arguments ---
              std::vector<std::string> args = {
                  "TS.CREATE", key,         "LABELS",      "channel",
                  channel_id,  "baseline",  baseline_pair, "polarization",
                  pol_pair,    "component", component // Add the final unique
                                                      // label
              };

              // --- 3. Execute the Command ---
              // Use the `command` method which accepts iterators for arguments
              try {
                redis.command(args.begin(), args.end());
                // std::cout << "Created key: " << key << std::endl; //
                // Uncomment for debugging
              } catch (const std::exception &e) {
                std::cerr << "Error creating key " << key << ": " << e.what()
                          << std::endl;
                // Handle error (e.g., key already exists, server down)
              }
            } // end component loop
          } // end pol_c_idx loop
        } // end pol_r_idx loop
      } // end bl_idx loop
    } // end ch_idx loop

    std::cout << "TimeSeries key pre-creation complete." << std::endl;
  }
  HighFive::File &file_;
  size_t element_count_;
  HighFive::DataSet vis_dataset_;
  HighFive::DataSet vis_seq_dataset_;
  HighFive::DataSet vis_missing_dataset_;
  std::vector<size_t> vis_dims_;
  std::unordered_map<int, int> antenna_map_;
  sw::redis::Redis redis;
  int NR_CHANNELS;
  int NR_BASELINES;
  int NR_POLARIZATIONS;
};

template <typename TVal, typename TVec>
class RedisEigendataWriter : public EigenWriter<TVal, TVec> {
public:
  RedisEigendataWriter()
      : val_element_count_(
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
    std::cout << "RedisEigendataWriter initialized with NR_CHANNELS: "
              << NR_CHANNELS << ", NR_POL: " << NR_POLARIZATIONS
              << ", NR_RECEIVERS" << NR_RECEIVERS << std::endl;
    create_all_timeseries_keys();
  }

  void write_eigendata_block(const TVal *val_data, const TVec *vec_data,
                             const int start_seq, const int end_seq) override {

    // NOTE: 'ts' (timestamp) needs to be passed or calculated here. Assuming
    // 'ts' is available. Example:
    long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

    LOG_INFO("writing eigendata block {} to {}", start_seq, end_seq);

    std::vector<std::string> madd_args = {"TS.MADD"};

    const int N = NR_RECEIVERS;

    // Reinterpret the template pointers to the underlying float/double types
    // Since TVal is a float array, reinterpret_cast is fine.
    const float *val_ptr = reinterpret_cast<const float *>(val_data);

    std::string ts_str = std::to_string(ts);

    for (int ch_idx = 0; ch_idx < NR_CHANNELS; ++ch_idx) {
      std::string channel_id = std::to_string(ch_idx);

      for (int pol_r_idx = 0; pol_r_idx < NR_POLARIZATIONS; ++pol_r_idx) {
        for (int pol_c_idx = 0; pol_c_idx < NR_POLARIZATIONS; ++pol_c_idx) {

          std::string pol_pair =
              std::to_string(pol_r_idx) + "-" + std::to_string(pol_c_idx);

          // === EIGENVALUE (val_data) ACCESS ===
          // TVal layout: [CH][POL_R][POL_C][N] -> 4D array
          size_t val_base_offset =
              ch_idx * (NR_POLARIZATIONS * NR_POLARIZATIONS * N) +
              pol_r_idx * (NR_POLARIZATIONS * N) + pol_c_idx * N;

          for (int k_idx = 0; k_idx < N; ++k_idx) {
            // --- 1. Data Access ---
            const float eigenvalue = val_ptr[val_base_offset + k_idx];

            // --- 2. Build TS.MADD Arg ---
            std::string k_id = std::to_string(k_idx);
            std::string key_prefix =
                "ts:ch:" + channel_id + ":p:" + pol_pair + ":k:" + k_id;

            madd_args.push_back(key_prefix + ":val");
            madd_args.push_back(ts_str);
            madd_args.push_back(std::to_string(eigenvalue));
          } // End EIGENVALUE k_idx loop
        }
      }
    }

    // --- 3. Execute TS.MADD ---
    if (madd_args.size() > 1) {
      redis.command(madd_args.begin(), madd_args.end());
    }
  }

  void flush() override {}

private:
  void create_all_timeseries_keys() {
    const int N = NR_RECEIVERS; // Matrix dimension
    // Track ALL N eigenvalues: "val"
    const std::vector<std::string> val_components = {"val"};

    std::cout << "Starting TimeSeries key pre-creation..." << std::endl;

    // Calculate total keys to be created
    int total_pol_pairs = NR_POLARIZATIONS * NR_POLARIZATIONS;
    int total_eigenvalue_keys =
        NR_CHANNELS * total_pol_pairs * N * val_components.size();

    std::cout << "Total keys to create: " << (total_eigenvalue_keys)
              << std::endl;

    for (int ch_idx = 0; ch_idx < NR_CHANNELS; ++ch_idx) {
      std::string channel_id = std::to_string(ch_idx);

      for (int pol_r_idx = 0; pol_r_idx < NR_POLARIZATIONS; ++pol_r_idx) {
        for (int pol_c_idx = 0; pol_c_idx < NR_POLARIZATIONS; ++pol_c_idx) {
          std::string pol_pair =
              std::to_string(pol_r_idx) + "-" + std::to_string(pol_c_idx);

          // === 1. EIGENVALUE KEYS (All N components) ===
          for (int k_idx = 0; k_idx < N; ++k_idx) {
            std::string k_id = std::to_string(k_idx);
            for (const auto &component : val_components) {

              // Key: ts:ch:<CH>:p:<P-P>:k:<K>:val
              std::string key = "ts:ch:" + channel_id + ":p:" + pol_pair +
                                ":k:" + k_id + ":" + component;

              std::vector<std::string> args = {
                  "TS.CREATE",    key,      "LABELS",    "channel", channel_id,
                  "polarization", pol_pair, "eigen_idx", k_id,      "component",
                  component};
              try {
                redis.command(args.begin(), args.end());
              } catch (const std::exception &e) {
                std::cerr << "Error creating key " << key << ": " << e.what()
                          << std::endl;
              }
            }
          } // End EIGENVALUE k_idx loop
        }
      }
    }
    std::cout << "TimeSeries key pre-creation complete." << std::endl;
  }
  size_t val_element_count_;
  size_t vec_element_count_;
  std::vector<size_t> eigen_dims_;
  sw::redis::Redis redis;
  int NR_CHANNELS;
  int NR_POLARIZATIONS;
  int NR_RECEIVERS;
};

template <typename T> class RedisFFTWriter : public FFTWriter<T> {
public:
  RedisFFTWriter()
      : element_count_(sizeof(T) /
                       sizeof(typename std::remove_all_extents<T>::type)),
        redis("tcp://127.0.0.1:6379") {
    fft_dims_ = get_array_dims<T>();

    NR_CHANNELS = fft_dims_[0];
    NR_POLARIZATIONS = fft_dims_[1];
    NR_RECEIVERS = fft_dims_[2];
    NR_FREQS = fft_dims_[3];
    DOWNSAMPLE_FACTOR = 32;
    std::cout << "RedisFFTWriter has NR_CHANNELS: " << NR_CHANNELS
              << ", NR_POL: " << NR_POLARIZATIONS
              << ", NR_RECEIVERS: " << NR_RECEIVERS
              << ", NR_FREQS: " << NR_FREQS << std::endl;
    create_all_timeseries_keys();
  }
  float complex_half_mag(const std::complex<__half> &c) {
    float re = __half2float(c.real());
    float im = __half2float(c.imag());
    return std::sqrt(re * re + im * im);
  }

  void write_fft_block(const T *fft_data, const int start_seq,
                       const int end_seq) override {

    long long ts = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::system_clock::now().time_since_epoch())
                       .count();

    LOG_INFO("writing fft block {} to {}", start_seq, end_seq);

    std::vector<std::string> madd_args = {"TS.MADD"};

    const int F = NR_FREQS;
    // There will be a little left over.
    const int NUM_FREQS_DOWNSAMPLED = F / DOWNSAMPLE_FACTOR;

    for (int ch = 0; ch < NR_CHANNELS; ++ch) {
      for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
        for (int rx = 0; rx < NR_RECEIVERS; ++rx) {

          for (int f = 0; f < NUM_FREQS_DOWNSAMPLED; ++f) {

            float power_sum = 0.0f;
            for (int i = 0; i < DOWNSAMPLE_FACTOR; ++i) {
              const auto &cval = fft_data[0][ch][pol][rx][f];
              float magnitude = complex_half_mag(cval);
              power_sum += magnitude;
            }

            int f_shifted =
                (f + NUM_FREQS_DOWNSAMPLED / 2) % NUM_FREQS_DOWNSAMPLED;
            std::string key = "ts:fft:ch:" + std::to_string(ch) +
                              ":p:" + std::to_string(pol) +
                              ":r:" + std::to_string(rx) +
                              ":f:" + std::to_string(f_shifted);

            madd_args.push_back(key);
            madd_args.push_back(std::to_string(ts));
            madd_args.push_back(std::to_string(power_sum / DOWNSAMPLE_FACTOR));
          }
        }
      }
    }

    if (madd_args.size() > 1) {
      redis.command(madd_args.begin(), madd_args.end());
    }
  }

  void flush() override {}

private:
  void create_all_timeseries_keys() {
    std::cout << "Pre-creating FFT TimeSeries keys..." << std::endl;

    int NUM_FREQS_OUT = NR_FREQS / DOWNSAMPLE_FACTOR;
    for (int ch = 0; ch < NR_CHANNELS; ++ch) {
      for (int pol = 0; pol < NR_POLARIZATIONS; ++pol) {
        for (int rx = 0; rx < NR_RECEIVERS; ++rx) {
          for (int f = 0; f < NR_FREQS_OUT; ++f) {

            std::string key = "ts:fft:ch:" + std::to_string(ch) +
                              ":p:" + std::to_string(pol) +
                              ":r:" + std::to_string(rx) +
                              ":f:" + std::to_string(f);

            std::vector<std::string> args = {"TS.CREATE",
                                             key,
                                             "LABELS",
                                             "channel",
                                             std::to_string(ch),
                                             "polarization",
                                             std::to_string(pol),
                                             "receiver",
                                             std::to_string(rx),
                                             "freq",
                                             std::to_string(f)};

            try {
              redis.command(args.begin(), args.end());
            } catch (const std::exception &e) {
              std::cerr << "Error creating key " << key << ": " << e.what()
                        << std::endl;
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
  int NR_RECEIVERS;
  int NR_FREQS;

  int DOWNSAMPLE_FACTOR;
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
    using beam_type = typename std::remove_all_extents_t<BeamT>;
    using arrival_type = typename std::remove_all_extents_t<ArrivalsT>;

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
    using beam_type = typename std::remove_all_extents_t<BeamT>;
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
    using arrival_type = typename std::remove_all_extents_t<ArrivalsT>;
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

    using beam_type = typename std::remove_all_extents_t<BeamT>;
    using arrival_type = typename std::remove_all_extents_t<ArrivalsT>;

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

    LOG_INFO("[InMemoryRawBeamWriter] Allocated space for {} blocks ({} + {} "
             "bytes each)",
             capacity, sizeof(BeamT), sizeof(ArrivalsT));
  }

  ~InMemoryBeamWriter() override {
    std::free(beam_buffer_);
    std::free(arrivals_buffer_);
    std::free(meta_buffer_);
    LOG_INFO("[InMemoryRawBeamWriter] Freed buffers.");
  }

  void write_beam_block(const BeamT *beam_data, const ArrivalsT *arrivals_data,
                        int start_seq, int end_seq) override {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();

    size_t write_index = (start_index_ + count_) % capacity_;
    if (count_ == capacity_) {
      // Overwrite oldest
      start_index_ = (start_index_ + 1) % capacity_;
      LOG_INFO(
          "[InMemoryRawBeamWriter] Buffer full, overwriting oldest block.");
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

    LOG_DEBUG("[InMemoryRawBeamWriter] Copied block {} in {} us.", write_index,
              us);
  }

  void flush() override { LOG_INFO("[InMemoryRawBeamWriter] Flush (no-op)."); }

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

template <typename T>
class UVFITSVisibilitiesWriter : public VisibilitiesWriter<T> {
public:
  using value_type = float;

  UVFITSVisibilitiesWriter(const std::string &filename, size_t n_chan,
                           size_t n_pol, size_t n_receivers, double ref_freq_hz,
                           double chan_width_hz, double integration_time_sec,
                           double ref_jd
                           // const std::vector<double> &antenna_xyz
                           )
      : filename_(filename), nant_(n_receivers), nchan_(n_chan), npol_(n_pol),
        nstokes_(4), // XX, YY, XY, YX
        ref_freq_(ref_freq_hz), dfreq_(chan_width_hz),
        int_time_sec_(integration_time_sec), ref_jd_(ref_jd), fptr_(nullptr),
        row_counter_(0) {
    //    if (antenna_xyz.size() != size_t(3 * nant))
    //    throw std::runtime_error("antenna_xyz size must be 3*nant");

    // antpos_ = antenna_xyz;

    // ------- Build baseline list -------
    for (int i = 0; i < nant_; i++)
      for (int j = i; j < nant_; j++) {
        bl1_.push_back(i);
        bl2_.push_back(j);
      }
    nbl_ = bl1_.size();

    // ------- Open FITS file -------
    int status = 0;
    if (fits_create_file(&fptr_, ("!" + filename_).c_str(), &status))
      error(status, "fits_create_file");

    if (fits_create_img(fptr_, FLOAT_IMG, 0, nullptr, &status))
      error(status, "fits_create_img");

    // ------- Primary HDU keywords -------
    fits_update_key(fptr_, TSTRING, "ORIGIN", (void *)"SIM", nullptr, &status);
    fits_update_key(fptr_, TDOUBLE, "CRVAL3", &ref_freq_, nullptr, &status);
    fits_update_key(fptr_, TDOUBLE, "CDELT3", &dfreq_, nullptr, &status);
    fits_update_key(fptr_, TINT, "NAXIS3", &nchan_, nullptr, &status);

    int stokes_start = -5; // XX
    int stokes_step = -1;
    fits_update_key(fptr_, TSTRING, "CTYPE4", (void *)"STOKES", nullptr,
                    &status);
    fits_update_key(fptr_, TINT, "CRVAL4", &stokes_start, nullptr, &status);
    fits_update_key(fptr_, TINT, "CDELT4", &stokes_step, nullptr, &status);
    fits_update_key(fptr_, TINT, "NAXIS4", &nstokes_, nullptr, &status);

    // --- UV_DATA table with new columns for start/end sequence ---
    const int NCOLS = 9; // Increased from 7 to 9
    char *ttype[NCOLS] = {
        (char *)"UU",   (char *)"VV",        (char *)"WW",
        (char *)"DATE", (char *)"TIME",      (char *)"BASELINE",
        (char *)"DATA", (char *)"START_SEQ", (char *)"END_SEQ"};

    int nvis = nstokes_ * nchan_;
    tform_data_ = std::to_string(nvis) + "C";

    char *tform[NCOLS] = {(char *)"1D",
                          (char *)"1D",
                          (char *)"1D",
                          (char *)"1D",
                          (char *)"1D",
                          (char *)"1J",
                          (char *)tform_data_.c_str(),
                          (char *)"1J",  // New: START_SEQ
                          (char *)"1J"}; // New: END_SEQ

    char *tunit[NCOLS] = {
        (char *)"SECONDS", (char *)"SECONDS", (char *)"SECONDS",
        (char *)"DAY",     (char *)"DAY",     (char *)"",
        (char *)"JY",      (char *)"",        (char *)""};

    if (fits_create_tbl(fptr_, BINARY_TBL, 0, NCOLS, ttype, tform, tunit,
                        "UV_DATA", &status))
      error(status, "fits_create_tbl");
  }

  void write_visibilities_block(const T *data, const int start_seq,
                                const int end_seq,
                                const int num_missing_packets,
                                const int num_total_packets) override {
    double center_seq = 0.5 * (start_seq + end_seq);
    double time_jd = ref_jd_ + center_seq * (int_time_sec_ / 86400.0);

    write_block_internal(data, time_jd, start_seq, end_seq);
  }

  ~UVFITSVisibilitiesWriter() { flush(); }
  void flush() override {
    int status = 0;
    fits_flush_file(fptr_, &status);
  }

private:
  void write_block_internal(const T *data, double time_jd, const int start_seq,
                            const int end_seq) {
    int status = 0;
    fits_movnam_hdu(fptr_, BINARY_TBL, (char *)"UV_DATA", 0, &status);

    std::vector<float> vis(2 * nstokes_ * nchan_);

    double jd_int = floor(time_jd);
    double jd_frac = time_jd - jd_int;

    // mapping of stokes index → (p1,p2)
    static const int s_p1[4] = {0, 1, 0, 1};
    static const int s_p2[4] = {0, 1, 1, 0};

    // per baseline
    for (int b = 0; b < nbl_; b++) {
      row_counter_++;
      fits_insert_rows(fptr_, row_counter_ - 1, 1, &status);

      int a1 = bl1_[b];
      int a2 = bl2_[b];

      // NOTE: antpos_ must be defined/filled for this to compile/work correctly
      // double uu = antpos_[3 * a2] - antpos_[3 * a1];
      // double vv = antpos_[3 * a2 + 1] - antpos_[3 * a1 + 1];
      // double ww = antpos_[3 * a2 + 2] - antpos_[3 * a1 + 2];
      // Using dummy values for demonstration since antpos_ is commented out
      double uu = 0.0, vv = 0.0, ww = 0.0;

      fits_write_col(fptr_, TDOUBLE, 1, row_counter_, 1, 1, &uu, &status);
      fits_write_col(fptr_, TDOUBLE, 2, row_counter_, 1, 1, &vv, &status);
      fits_write_col(fptr_, TDOUBLE, 3, row_counter_, 1, 1, &ww, &status);
      fits_write_col(fptr_, TDOUBLE, 4, row_counter_, 1, 1, &jd_int, &status);
      fits_write_col(fptr_, TDOUBLE, 5, row_counter_, 1, 1, &jd_frac, &status);

      int blcode = 256 * (a1 + 1) + (a2 + 1);
      fits_write_col(fptr_, TINT, 6, row_counter_, 1, 1, &blcode, &status);

      // New columns (8 and 9)
      fits_write_col(fptr_, TINT, 8, row_counter_, 1, 1, (void *)&start_seq,
                     &status);
      fits_write_col(fptr_, TINT, 9, row_counter_, 1, 1, (void *)&end_seq,
                     &status);

      // reorder visibilities into AIPS stokes order
      for (int s = 0; s < nstokes_; s++) {
        int p1 = s_p1[s];
        int p2 = s_p2[s];

        for (int c = 0; c < nchan_; c++) {

          size_t out_idx = (s * nchan_ + c) * 2;
          vis[out_idx + 0] = data[0][c][b][p1][p2][0];
          vis[out_idx + 1] = data[0][c][b][p1][p2][1];
        }
      }

      fits_write_col(fptr_, TFLOAT, 7, row_counter_, 1, vis.size(), vis.data(),
                     &status);
      if (status)
        error(status, "fits_write_col(DATA)");
    }
  }

  void error(int status, const char *msg) {
    // fits_report_error(stderr, status); // Disabled for cleaner output here
    throw std::runtime_error("CFITSIO: " + std::string(msg));
  }

  // state
  std::string filename_;
  fitsfile *fptr_;
  std::string tform_data_;

  int nant_, nchan_, npol_, nstokes_, nbl_;
  double ref_freq_, dfreq_;
  double int_time_sec_, ref_jd_;

  std::vector<double> antpos_;
  std::vector<int> bl1_, bl2_;
  long row_counter_;
};

template <typename T>
class MSVisibilitiesWriter : public VisibilitiesWriter<T> {
public:
  // T is expected to be float[CH][BL][POL][POL][CPLX]
  // We derive dimensions from T at compile time/runtime
  static constexpr size_t NUM_CHANNELS = std::extent<T, 0>::value;
  static constexpr size_t NUM_BASELINES = std::extent<T, 1>::value;
  // Assuming [POL][POL][CPLX] flattens to 4 complex numbers
  static constexpr size_t NUM_CORRELATIONS = 4;

  MSVisibilitiesWriter(
      const std::string &filename,
      const std::unordered_map<int, int> *antenna_map = nullptr)
      : current_row_(0) {
    // 1. Setup the MS Definition
    casacore::TableDesc td = casacore::MS::requiredTableDesc();
    casacore::MS::addColumnToDesc(
        td, casacore::MS::DATA); // Add DATA column (Standard MS doesn't imply
                                 // DATA/CORRECTED_DATA by default)

    casacore::SetupNewTable newTab(filename, td, casacore::Table::New);

    // 2. Create the MeasurementSet
    ms_ = std::make_unique<casacore::MeasurementSet>(newTab);
    ms_->createDefaultSubtables(casacore::Table::New);
    // 3. Initialize Subtables (Antenna, Field, SpectralWindow, etc.)
    setup_subtables();

    // 4. Initialize Columns accessors
    // We use MSMainColumns for convenience, or individual column accessors
    msc_ = std::make_unique<casacore::MSColumns>(*ms_);

    // Handle Antenna Map
    if (antenna_map && !antenna_map->empty()) {
      this->antenna_map_ = *antenna_map;
    } else {
      generate_identity_map();
    }

    // Pre-calculate baseline pairs (Ant1, Ant2) to save time in the loop
    calculate_baseline_pairs();
  }

  ~MSVisibilitiesWriter() {
    if (ms_) {
      ms_->flush();
    }
  }

  void write_visibilities_block(const T *data, const int start_seq,
                                const int end_seq,
                                const int num_missing_packets,
                                const int num_total_packets) override {
    // Add rows for this block (1 timestamp * NR_BASELINES)
    size_t start_row = current_row_;
    ms_->addRow(NUM_BASELINES);

    // Pointers/Refs for raw data
    // T is float[CH][BL][POL][POL][2] (Real/Imag)
    // We cast the raw pointer to complex<float> for easier iteration
    // Shape becomes: [CH][BL][4] complex elements
    const auto *complex_data =
        reinterpret_cast<const std::complex<float> *>(data);

    // Placeholder time: using sequence number as seconds for now.
    // In reality, this should be MJD Seconds.
    //    double timestamp = static_cast<double>(start_seq);
    constexpr double UNIX_TO_MJD_SECONDS = 40587.0 * 86400.0;
    double unix_timestamp =
        std::chrono::duration<double>(
            std::chrono::system_clock::now().time_since_epoch())
            .count();
    for (size_t bl = 0; bl < NUM_BASELINES; ++bl) {
      size_t row_idx = start_row + bl;
      auto &pair = baseline_pairs_[bl];

      // 1. Metadata Columns
      msc_->antenna1().put(row_idx, pair.first);
      msc_->antenna2().put(row_idx, pair.second);
      msc_->time().put(row_idx, unix_timestamp + UNIX_TO_MJD_SECONDS);
      msc_->interval().put(row_idx, 1.0); // 1 second integration placeholder
      msc_->scanNumber().put(row_idx, 1);
      msc_->fieldId().put(row_idx, 0); // Pointing to index 0 in Field table
      msc_->dataDescId().put(row_idx,
                             0); // Pointing to index 0 in DataDesc table

      // 2. UVW (Placeholder: Blank/Zero)
      casacore::Vector<double> uvw(3, 0.0);
      msc_->uvw().put(row_idx, uvw);

      // 3. Data
      // Casacore expects Matrix<Complex> of shape (NumPol, NumChan)
      casacore::Matrix<casacore::Complex> vis_matrix(NUM_CORRELATIONS,
                                                     NUM_CHANNELS);

      for (size_t ch = 0; ch < NUM_CHANNELS; ++ch) {
        // Input data layout calculation:
        // Global Index = ch * (NUM_BASELINES * 4) + bl * 4 + pol
        size_t base_idx =
            ch * (NUM_BASELINES * NUM_CORRELATIONS) + bl * NUM_CORRELATIONS;

        for (size_t pol = 0; pol < NUM_CORRELATIONS; ++pol) {
          std::complex<float> val = complex_data[base_idx + pol];
          vis_matrix(pol, ch) = casacore::Complex(val.real(), val.imag());
        }
      }

      msc_->data().put(row_idx, vis_matrix);

      // Flags/Sigmas (Standard defaults)
      msc_->flagRow().put(row_idx, false);
      // Initialize weights to 1.0
      casacore::Vector<float> weight(NUM_CORRELATIONS, 1.0f);
      msc_->weight().put(row_idx, weight);
      msc_->sigma().put(row_idx, weight);
    }

    current_row_ += NUM_BASELINES;
  }

  void flush() override {
    if (ms_)
      ms_->flush();
  }

private:
  std::unique_ptr<casacore::MeasurementSet> ms_;
  std::unique_ptr<casacore::MSColumns> msc_;
  size_t current_row_;
  std::unordered_map<int, int> antenna_map_;
  std::vector<std::pair<int, int>> baseline_pairs_;

  void calculate_baseline_pairs() {
    size_t nr_antennas =
        static_cast<size_t>((std::sqrt(1 + 8 * NUM_BASELINES) - 1) / 2);
    baseline_pairs_.reserve(NUM_BASELINES);

    // Same triangular logic as HDF5 writer
    for (size_t ant2 = 0; ant2 < nr_antennas; ++ant2) {
      for (size_t ant1 = 0; ant1 <= ant2; ++ant1) {
        baseline_pairs_.emplace_back(antenna_map_[ant1], antenna_map_[ant2]);
      }
    }
  }

  void generate_identity_map() {
    for (int i = 0; i < 256; i++) {
      antenna_map_[i] = i;
    }
  }

  void setup_subtables() {
    // --- 1. POLARIZATION ---
    // Setup 4 correlations (XX, XY, YX, YY)
    {
      casacore::MSPolarization &polTable = ms_->polarization();
      polTable.addRow(1);
      casacore::MSPolarizationColumns polCols(polTable);

      casacore::Vector<int> corrType(4);
      corrType[0] = 9;  // XX
      corrType[1] = 10; // XY
      corrType[2] = 11; // YX
      corrType[3] = 12; // YY

      casacore::Matrix<int> corrProduct(2, 4);
      // Map receptor pairs to correlations (placeholder logic)
      corrProduct = 0;

      polCols.numCorr().put(0, 4);
      polCols.corrType().put(0, corrType);
      polCols.corrProduct().put(0, corrProduct);
    }

    // --- 2. SPECTRAL WINDOW ---
    {
      casacore::MSSpectralWindow &spwTable = ms_->spectralWindow();
      spwTable.addRow(1);
      casacore::MSSpWindowColumns spwCols(spwTable);

      spwCols.numChan().put(0, NUM_CHANNELS);
      spwCols.name().put(0, "Subband_0");

      // Frequencies (Placeholder: 1GHz + 1MHz channels)
      casacore::Vector<double> chanFreqs(NUM_CHANNELS);
      casacore::Vector<double> chanWidths(NUM_CHANNELS);
      double start_freq = 1.0e9;
      double width = 1.0e6;

      for (size_t i = 0; i < NUM_CHANNELS; ++i) {
        chanFreqs[i] = start_freq + i * width;
        chanWidths[i] = width;
      }

      spwCols.chanFreq().put(0, chanFreqs);
      spwCols.chanWidth().put(0, chanWidths);
      spwCols.resolution().put(0, chanWidths);
      spwCols.refFrequency().put(0, start_freq);
      spwCols.totalBandwidth().put(0, width * NUM_CHANNELS);
    }

    // --- 3. DATA DESCRIPTION ---
    // Links Pol and SpW
    {
      casacore::MSDataDescription &ddTable = ms_->dataDescription();
      ddTable.addRow(1);
      casacore::MSDataDescColumns ddCols(ddTable);
      ddCols.spectralWindowId().put(0, 0);
      ddCols.polarizationId().put(0, 0);
    }

    // --- 4. ANTENNA ---
    {
      size_t nr_antennas =
          static_cast<size_t>((std::sqrt(1 + 8 * NUM_BASELINES) - 1) / 2);
      casacore::MSAntenna &antTable = ms_->antenna();
      antTable.addRow(nr_antennas);
      casacore::MSAntennaColumns antCols(antTable);

      for (size_t i = 0; i < nr_antennas; ++i) {
        antCols.name().put(i, "ANT" + std::to_string(antenna_map_[i]));
        antCols.station().put(i, "STATION" + std::to_string(antenna_map_[i]));
        antCols.mount().put(i, "ALT-AZ");
        // Placeholder Position (ITRF)
        casacore::Vector<double> pos(3, 0.0);
        antCols.position().put(i, pos);
      }
    }

    // --- 5. FIELD (Placeholder) ---
    {
      casacore::MSField &fieldTable = ms_->field();
      fieldTable.addRow(1);
      casacore::MSFieldColumns fieldCols(fieldTable);

      fieldCols.name().put(0, "PLACEHOLDER_SOURCE");
      fieldCols.code().put(0, "NONE");

      // Direction (RA/DEC) - Zero for now
      casacore::Matrix<double> dir(2, 1, 0.0);
      fieldCols.delayDir().put(0, dir);
      fieldCols.phaseDir().put(0, dir);
      fieldCols.referenceDir().put(0, dir);
    }
  }
};
