#pragma once
#include <cuda_fp16.h>
#include <cutensor.h>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct TensorMeta {
  std::vector<int> modes;
  std::vector<int64_t> extents;
  cutensorTensorDescriptor_t desc;
  size_t elements;
  size_t sizeBytes;

  TensorMeta() : elements(0), sizeBytes(0) {}
};

struct PermutationOp {
  cutensorOperationDescriptor_t desc;
  cutensorPlan_t plan;
  std::string fromTensor;
  std::string toTensor;

  PermutationOp() = default;
  PermutationOp(const std::string &from, const std::string &to)
      : fromTensor(from), toTensor(to) {}
};

class CutensorSetup {
public:
  // Constructor with extent mapping
  CutensorSetup(const std::unordered_map<int, int64_t> &extentMap,
                cutensorDataType_t dataType = CUTENSOR_R_16F,
                uint32_t alignment = 128);

  // Destructor
  ~CutensorSetup();

  // Add a tensor with given modes and name
  void addTensor(const std::vector<int> &modes, const std::string &name);

  // Add a permutation operation between two tensors
  void addPermutation(const std::string &fromTensorName,
                      const std::string &toTensorName,
                      cutensorComputeDescriptor_t computeType,
                      const std::string &opName);

  // Get tensor metadata
  const TensorMeta *getTensor(const std::string &name) const;

  // Get permutation operation
  const PermutationOp *getPermutation(const std::string &name) const;

  void runPermutation(const std::string &name, const __half &alpha,
                      const __half *d_in, __half *d_out, cudaStream_t stream);
  void runPermutation(const std::string &name, const float &alpha,
                      const float *d_in, float *d_out, cudaStream_t stream);
  void runPermutation(const std::string &name, const float &alpha,
                      const void *d_in, void *d_out, cudaStream_t stream);
  // Get cuTENSOR handle
  cutensorHandle_t getHandle() const { return handle; }

  // Utility: calculate total elements for a tensor
  size_t calculateElements(const std::vector<int> &modes) const;

  // Utility: calculate memory size for a tensor
  size_t calculateSizeBytes(const std::vector<int> &modes) const;

private:
  cutensorHandle_t handle;
  cutensorPlanPreference_t planPref;
  cutensorDataType_t dataType;
  uint32_t alignment;
  std::unordered_map<int, int64_t> extentMap;

  std::unordered_map<std::string, std::unique_ptr<TensorMeta>> tensors;
  std::unordered_map<std::string, std::unique_ptr<PermutationOp>> ops;

  // Helper function to create tensor descriptor
  void createTensorDescriptor(TensorMeta &meta, const std::vector<int> &modes);

  // Helper function to get element size based on data type
  size_t getElementSize() const;
};
