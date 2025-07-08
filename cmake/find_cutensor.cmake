

if(NOT DEFINED ENV{CUTENSOR_ROOT} AND NOT DEFINED CUTENSOR_ROOT)
  message(FATAL_ERROR "CUTENSOR_ROOT not set!")
else()
  if(DEFINED ENV{CUTENSOR_ROOT})
    set(CUTENSOR_ROOT "$ENV{CUTENSOR_ROOT}")
  endif()
  message("-- Looking for cuTENSOR in ${CUTENSOR_ROOT}")
  if(NOT EXISTS ${CUTENSOR_ROOT})
    message(FATAL_ERROR "Cannot find CUTENSOR_ROOT")
  endif()
endif()

if(NOT TARGET cutensor)
  add_library(cutensor SHARED IMPORTED)
  set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
      set(LIB_DIR "/lib/12")
  set_target_properties(cutensor PROPERTIES
    IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
    IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
endif()

