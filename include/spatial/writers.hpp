#pragma once

#include <array>
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
    // Write beam data
    auto current_size = beam_dataset_.getDimensions()[0];
    std::vector<size_t> new_dims = {current_size + 1};
    new_dims.insert(new_dims.end(), beam_dims_.begin(), beam_dims_.end());
    beam_dataset_.resize(new_dims);

    std::vector<size_t> beam_offset = {current_size};
    beam_offset.insert(beam_offset.end(), beam_dims_.size(), 0);
    std::vector<size_t> beam_count = {1};
    beam_count.insert(beam_count.end(), beam_dims_.begin(), beam_dims_.end());

    using beam_type = std::remove_all_extents<BeamT>::type;
    beam_dataset_.select(beam_offset, beam_count).write(beam_data);

    // Write arrivals data
    auto arrivals_size = arrivals_dataset_.getDimensions()[0];
    std::vector<size_t> arrivals_new_dims = {arrivals_size + 1};
    arrivals_new_dims.insert(arrivals_new_dims.end(), arrivals_dims_.begin(),
                             arrivals_dims_.end());
    std::vector<size_t> arrivals_offset = {arrivals_size};
    arrivals_offset.insert(arrivals_offset.end(), arrivals_dims_.size(), 0);
    std::vector<size_t> arrivals_count = {1};
    arrivals_count.insert(arrivals_count.end(), arrivals_dims_.begin(),
                          arrivals_dims_.end());

    arrivals_dataset_.resize(arrivals_new_dims);

    arrivals_dataset_.select(arrivals_offset, arrivals_count)
        .write_raw(arrivals_data);

    // Write sequence numbers
    auto seq_size = beam_seq_dataset_.getDimensions()[0];
    beam_seq_dataset_.resize({seq_size + 1, 2});
    std::vector<int> seq_nums = {start_seq, end_seq};
    beam_seq_dataset_.select({seq_size, 0}, {1, 2}).write_raw(seq_nums.data());
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
