
#include "spatial/tensor.hpp"
#include <algorithm>
#include <cuda_fp16.h>
#include <cutensor.h>
#include <iostream>
#include <stdexcept>

void checkCutensorStatus(cutensorStatus_t status, const char *msg) {
  if (status != CUTENSOR_STATUS_SUCCESS) {
    std::cerr << "cuTENSOR error at " << msg << ": "
              << cutensorGetErrorString(status) << std::endl;
    exit(EXIT_FAILURE);
  }
}

CutensorSetup::CutensorSetup(const std::unordered_map<int, int64_t> &extentMap,
                             cutensorDataType_t dataType, uint32_t alignment)
    : extentMap(extentMap), dataType(dataType), alignment(alignment) {

  // Initialize cuTENSOR handle
  cutensorStatus_t status = cutensorCreate(&handle);
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create cuTENSOR handle");
  }

  // Create plan preference with default algorithm
  status = cutensorCreatePlanPreference(
      handle, &planPref, CUTENSOR_ALGO_DEFAULT, CUTENSOR_JIT_MODE_NONE);
  if (status != CUTENSOR_STATUS_SUCCESS) {
    cutensorDestroy(handle);
    throw std::runtime_error("Failed to create plan preference");
  }
}

CutensorSetup::~CutensorSetup() {
  // Clean up operations (plans and descriptors)
  for (auto &op : ops) {
    // Note: cuTENSOR handles cleanup of descriptors and plans automatically
    // when the handle is destroyed, but explicit cleanup is good practice
  }

  // Clean up tensors
  for (auto &tensor : tensors) {
    // Tensor descriptors are cleaned up when handle is destroyed
  }

  // Destroy cuTENSOR handle
  if (handle) {
    cutensorDestroy(handle);
  }
}

void CutensorSetup::addTensor(const std::vector<int> &modes,
                              const std::string &name) {
  if (tensors.find(name) != tensors.end()) {
    throw std::runtime_error("Tensor with name '" + name + "' already exists");
  }
  std::vector<int> reversed_modes = modes;
  std::reverse(reversed_modes.begin(), reversed_modes.end());
  auto meta = std::make_unique<TensorMeta>();
  meta->modes = reversed_modes;
  // Calculate extents based on modes
  meta->extents.reserve(reversed_modes.size());
  for (int mode : reversed_modes) {
    auto it = extentMap.find(mode);
    if (it == extentMap.end()) {
      throw std::runtime_error("Mode '" + std::to_string(mode) +
                               "' not found in extent map");
    }
    meta->extents.push_back(it->second);
  }

  // Calculate elements and size
  meta->elements = calculateElements(reversed_modes);
  meta->sizeBytes = calculateSizeBytes(reversed_modes);

  // Create tensor descriptor
  createTensorDescriptor(*meta, reversed_modes);

  std::cout << "Tensor " << name << " created with " << meta->elements
            << " elements and size " << meta->sizeBytes << " bytes. Modes ";
  for (auto ex : meta->modes) {
    std::cout << ex;
  }
  std::cout << std::endl;

  // Store the tensor
  tensors[name] = std::move(meta);
}

void CutensorSetup::addPermutation(const std::string &fromTensorName,
                                   const std::string &toTensorName,
                                   cutensorComputeDescriptor_t computeType,
                                   const std::string &opName) {

  if (ops.find(opName) != ops.end()) {
    throw std::runtime_error("Operation with name '" + opName +
                             "' already exists");
  }

  // Check if tensors exist
  auto fromIt = tensors.find(fromTensorName);
  auto toIt = tensors.find(toTensorName);

  if (fromIt == tensors.end()) {
    throw std::runtime_error("Source tensor '" + fromTensorName +
                             "' not found");
  }
  if (toIt == tensors.end()) {
    throw std::runtime_error("Destination tensor '" + toTensorName +
                             "' not found");
  }

  auto op = std::make_unique<PermutationOp>(fromTensorName, toTensorName);

  // Create permutation operation descriptor
  cutensorStatus_t status = cutensorCreatePermutation(
      handle, &(op->desc), fromIt->second->desc, fromIt->second->modes.data(),
      CUTENSOR_OP_IDENTITY, toIt->second->desc, toIt->second->modes.data(),
      computeType);

  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create permutation operation");
  }

  // Create execution plan
  status = cutensorCreatePlan(handle, &(op->plan), op->desc, planPref, 0);
  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create execution plan");
  }

  std::cout << "Created permutation from " << fromTensorName << " to "
            << toTensorName << " with " << fromIt->second->elements << ", "
            << toIt->second->elements << " elements and shapes (";

  for (auto &item : fromIt->second->modes) {
    std::cout << extentMap[item] << ",";
  }
  std::cout << ") and (";

  for (const auto &item : toIt->second->modes) {
    std::cout << extentMap[item] << ",";
  }
  std::cout << ")\n";

  // Store the operation
  ops[opName] = std::move(op);
}

const TensorMeta *CutensorSetup::getTensor(const std::string &name) const {
  auto it = tensors.find(name);
  if (it == tensors.end()) {
    return nullptr;
  }
  return it->second.get();
}

const PermutationOp *
CutensorSetup::getPermutation(const std::string &name) const {
  auto it = ops.find(name);
  if (it == ops.end()) {

    std::cerr << "Could not find permutation " << name << "!\n";
    return nullptr;
  }
  return it->second.get();
}

void CutensorSetup::runPermutation(const std::string &name, const __half &alpha,
                                   const __half *d_in, __half *d_out,
                                   cudaStream_t stream) {
  const PermutationOp *perm = getPermutation(name);
  std::cout << "Running permutation " << name << "..." << std::endl;

  checkCutensorStatus(
      cutensorPermute(handle, perm->plan, &alpha, d_in, d_out, stream),
      "permutation __half");
}

void CutensorSetup::runPermutation(const std::string &name, const float &alpha,
                                   const float *d_in, float *d_out,
                                   cudaStream_t stream) {
  const PermutationOp *perm = getPermutation(name);
  std::cout << "Running permutation " << name << "..." << std::endl;
  checkCutensorStatus(
      cutensorPermute(handle, perm->plan, &alpha, d_in, d_out, stream),
      "permutation float");
}

void CutensorSetup::runPermutation(const std::string &name, const float &alpha,
                                   const void *d_in, void *d_out,
                                   cudaStream_t stream) {
  const PermutationOp *perm = getPermutation(name);

  checkCutensorStatus(
      cutensorPermute(handle, perm->plan, &alpha, d_in, d_out, stream),
      "permutation void");
}

size_t CutensorSetup::calculateElements(const std::vector<int> &modes) const {
  size_t elements = 1;
  for (int mode : modes) {
    auto it = extentMap.find(mode);
    if (it != extentMap.end()) {
      elements *= it->second;
    }
  }
  return elements;
}

size_t CutensorSetup::calculateSizeBytes(const std::vector<int> &modes) const {
  return calculateElements(modes) * getElementSize();
}

void CutensorSetup::createTensorDescriptor(TensorMeta &meta,
                                           const std::vector<int> &modes) {
  cutensorStatus_t status = cutensorCreateTensorDescriptor(
      handle, &meta.desc, static_cast<int>(modes.size()), meta.extents.data(),
      nullptr, // strides (nullptr for default)
      dataType, alignment);

  if (status != CUTENSOR_STATUS_SUCCESS) {
    throw std::runtime_error("Failed to create tensor descriptor");
  }
}

size_t CutensorSetup::getElementSize() const {
  switch (dataType) {
  case CUTENSOR_R_16F:
    return sizeof(__half);
  case CUTENSOR_R_32F:
    return sizeof(float);
  case CUTENSOR_R_64F:
    return sizeof(double);
  case CUTENSOR_C_16F:
    return 2 * sizeof(__half);
  case CUTENSOR_C_32F:
    return 2 * sizeof(float);
  case CUTENSOR_C_64F:
    return 2 * sizeof(double);
  default:
    return sizeof(__half); // Default fallback
  }
}
