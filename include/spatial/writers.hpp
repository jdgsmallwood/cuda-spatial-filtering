#pragma once

#include <condition_variable>
#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <memory>
#include <string>
#include <thread>
#include <vector>

template <typename BeamT, typename ArrivalsT> class BeamWriter {
public:
  virtual ~BeamWriter() = default;
  virtual void write_beam_block(const BeamT *beam_data,
                                const ArrivalsT *arrivals_data,
                                size_t beam_element_count,
                                size_t arrivals_element_count, int start_seq,
                                int end_seq) = 0;
  virtual void flush() = 0;
};

template <typename T> class VisibilitiesWriter {
public:
  virtual ~VisibilitiesWriter() = default;
  virtual void write_visibilities_block(const T *data, size_t element_count,
                                        int start_seq, int end_seq) = 0;
  virtual void flush() = 0;
};

template <typename BeamT, typename ArrivalsT>
class HDF5BeamWriter : public BeamWriter<BeamT, ArrivalsT> {
public:
  HDF5BeamWriter(HighFive::File &file) : file_(file) {

    typename beam_type = std::remove_extent<BeamT>::type;
    typename arrival_type = bool;
    beam_element_count_ = sizeof(BeamT) / sizeof(beam_type);
    arrivals_element_count_ = sizeof(ArrivalsT) / sizeof(bool);
    // Create beam dataset
    HighFive::DataSpace beam_space(
        {0, beam_element_count_},
        {HighFive::DataSpace::UNLIMITED, beam_element_count_});
    HighFive::DataSetCreateProps beam_props;
    beam_props.add(
        HighFive::Chunking(std::vector<hsize_t>{1, beam_element_count_}));
    beam_dataset_ =
        file_.createDataSet<beam_type>("beam_data", beam_space, beam_props);

    // Create arrivals dataset
    HighFive::DataSpace arrivals_space(
        {0, arrivals_element_count_},
        {HighFive::DataSpace::UNLIMITED, arrivals_element_count_});
    HighFive::DataSetCreateProps arrivals_props;
    arrivals_props.add(
        HighFive::Chunking(std::vector<hsize_t>{1, arrivals_element_count_}));
    arrivals_dataset_ =
        file_.createDataSet<bool>("arrivals", arrivals_space, arrivals_props);

    // Create sequence number dataset
    beam_seq_dataset_ = file_.createDataSet<int>(
        "beam_seq_nums",
        HighFive::DataSpace({0, 2}, {HighFive::DataSpace::UNLIMITED, 2}));
  }

  void write_beam_block(const BeamT *beam_data, const bool *arrivals_data,
                        int start_seq, int end_seq) override {
    // Write beam data
    auto current_size = beam_dataset_.getDimensions()[0];
    beam_dataset_.resize({current_size + 1, beam_element_count_});
    std::vector<BeamT> beam_vec(beam_data, beam_data + beam_element_count);
    beam_dataset_.select({current_size, 0}, {1, beam_element_count_})
        .write(beam_vec);

    // Write arrivals data
    auto arrivals_size = arrivals_dataset_.getDimensions()[0];
    arrivals_dataset_.resize({arrivals_size + 1, arrivals_element_count_});
    std::vector<bool> arrivals_vec(arrivals_data,
                                   arrivals_data + arrivals_element_count);
    arrivals_dataset_.select({arrivals_size, 0}, {1, arrivals_element_count_})
        .write(arrivals_vec);

    // Write sequence numbers
    auto seq_size = beam_seq_dataset_.getDimensions()[0];
    beam_seq_dataset_.resize({seq_size + 1, 2});
    std::vector<int> seq_nums = {start_seq, end_seq};
    beam_seq_dataset_.select({seq_size, 0}, {1, 2}).write(seq_nums);
  }

  void flush() override { file_.flush(); }

private:
  HighFive::File &file_;
  size_t beam_element_count_;
  size_t arrivals_element_count_;
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
                       sizeof(typename std::remove_extent<T>::type)) {
    HighFive::DataSpace vis_space(
        {0, element_count_}, {HighFive::DataSpace::UNLIMITED, element_count_});
    HighFive::DataSetCreateProps props;
    props.add(HighFive::Chunking(std::vector<hsize_t>{1, element_count_}));
    // I imagine createDataSet needs a primitive type
    vis_dataset_ = file_.createDataSet<T>("visibilities", vis_space, props);
    vis_seq_dataset_ = file_.createDataSet<int>(
        "vis_seq_nums",
        HighFive::DataSpace({0, 2}, {HighFive::DataSpace::UNLIMITED, 2}));
  }

  void write_visibilities_block(const T *data, size_t element_count,
                                int start_seq, int end_seq) override {
    auto current_size = vis_dataset_.getDimensions()[0];
    vis_dataset_.resize({current_size + 1, element_count_});

    std::vector<T> data_vec(data, data + element_count);
    vis_dataset_.select({current_size, 0}, {1, element_count_}).write(data_vec);

    auto seq_size = vis_seq_dataset_.getDimensions()[0];
    vis_seq_dataset_.resize({seq_size + 1, 2});
    std::vector<int> seq_nums = {start_seq, end_seq};
    vis_seq_dataset_.select({seq_size, 0}, {1, 2}).write(seq_nums);
  }

  void flush() override { file_.flush(); }

private:
  HighFive::File &file_;
  size_t element_count_;
  HighFive::DataSet vis_dataset_;
  HighFive::DataSet vis_seq_dataset_;
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
