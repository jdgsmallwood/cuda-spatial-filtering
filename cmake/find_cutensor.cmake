set(_cutensor_hint_roots "")

if(DEFINED CUTENSOR_ROOT)
  list(APPEND _cutensor_hint_roots "${CUTENSOR_ROOT}")
endif()
if(DEFINED ENV{CUTENSOR_ROOT})
  list(APPEND _cutensor_hint_roots "$ENV{CUTENSOR_ROOT}")
endif()
if(DEFINED CUDAToolkit_LIBRARY_ROOT)
  list(APPEND _cutensor_hint_roots "${CUDAToolkit_LIBRARY_ROOT}")
endif()
if(DEFINED CUDAToolkit_ROOT_DIR)
  list(APPEND _cutensor_hint_roots "${CUDAToolkit_ROOT_DIR}")
endif()
if(DEFINED CUDAToolkit_BIN_DIR)
  cmake_path(GET CUDAToolkit_BIN_DIR PARENT_PATH _cuda_toolkit_root)
  list(APPEND _cutensor_hint_roots "${_cuda_toolkit_root}")
endif()
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND _cutensor_hint_roots "$ENV{CONDA_PREFIX}")
endif()

find_library(CUTENSOR_LIBRARY
  NAMES cutensor libcutensor.so
  HINTS ${_cutensor_hint_roots}
  PATH_SUFFIXES lib lib64 targets/x86_64-linux/lib
)

if(NOT CUTENSOR_LIBRARY)
  find_library(CUTENSOR_LIBRARY
    NAMES cutensor libcutensor.so
    PATHS
      /usr/lib/x86_64-linux-gnu
      /usr/local/cuda/lib64
      /opt/conda/lib
      /opt/conda/targets/x86_64-linux/lib
  )
endif()

if(NOT CUTENSOR_LIBRARY)
  message(FATAL_ERROR "Could not find libcutensor.so. Set CUTENSOR_ROOT if it is installed in a non-standard location.")
endif()

cmake_path(GET CUTENSOR_LIBRARY PARENT_PATH CUTENSOR_LIBRARY_DIR)
message(STATUS "Using cuTENSOR library at ${CUTENSOR_LIBRARY}")

if(NOT TARGET cutensor)
  add_library(cutensor SHARED IMPORTED)
  set_target_properties(cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_LIBRARY}"
  )
endif()
