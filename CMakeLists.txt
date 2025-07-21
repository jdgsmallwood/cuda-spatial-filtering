# Works with 3.15 and tested through 4.0
cmake_minimum_required(VERSION 3.9...4.0)

# Project name and a few useful settings. Other commands can pick up the results
project(
  CUDASpatialFiltering
  VERSION 0.1
  DESCRIPTION "CUDA implementation of spatial filtering algorithms."
  LANGUAGES CUDA CXX)

# Only do these if this is the main project, and not if it is included through add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)

  # Optionally set things like CMAKE_CXX_STANDARD, CMAKE_POSITION_INDEPENDENT_CODE here

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_EXTENSIONS OFF)

  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Save user choice in another variable
  set(USER_BUILD_TESTING ${BUILD_TESTING})

  # Disable testing for dependencies forcibly
  set(BUILD_TESTING OFF CACHE BOOL "Disable tests for dependencies" FORCE)

  find_package(CUDAToolkit)
  # Testing only available if this is the main app
  # Note this needs to be done in the main CMakeLists
  # since it calls enable_testing, which must be in the
  # main CMakeLists.
  include(CTest)
include(cmake/find_cutensor.cmake)

  # Docs only available if this is the main app
  find_package(Doxygen)
  if(Doxygen_FOUND)
    add_subdirectory(docs)
  else()
    message(STATUS "Doxygen not found, not building docs")
  endif()
endif()
message(STATUS "install prefix ${CMAKE_INSTALL_PREFIX}")
message(STATUS "install include dir: ${CMAKE_INSTALL_INCLUDEDIR}$")
# FetchContent added in CMake 3.11, downloads during the configure step
# FetchContent_MakeAvailable was added in CMake 3.14; simpler usage
include(FetchContent)

# Accumulator library
# This is header only, so could be replaced with git submodules or FetchContent

# Formatting library
#FetchContent_Declare(
#  fmtlib
#  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#  GIT_TAG 5.3.0)
#FetchContent_MakeAvailable(fmtlib)
# Adds fmt::fmt

# The compiled library code is here
add_subdirectory(src)

# The executable code is here
if(${USER_BUILD_TESTING} STREQUAL "OFF")
add_subdirectory(apps)
endif()

message(STATUS "Including dependencies...")
message(STATUS "cudawrappers...")
#add_subdirectory(${PROJECT_SOURCE_DIR}/extern/cudawrappers)
FetchContent_Declare(
  cudawrappers
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/cudawrappers
)

FetchContent_MakeAvailable(cudawrappers)


FetchContent_Declare(
  xtl
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/xtl
)
FetchContent_MakeAvailable(xtl)

FetchContent_Declare(
  xtensor
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/xtensor
)
FetchContent_MakeAvailable(xtensor)
message(STATUS "CMAKE_SOURCE: ${CMAKE_SOURCE_DIR}")

FetchContent_Declare(
  tcc
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/tcc
)
FetchContent_MakeAvailable(tcc)

target_include_directories(tcc PRIVATE ${tcc_SOURCE_DIR})
message(STATUS "ccglib")
FetchContent_Declare(
  ccglib
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/ccglib
)
FetchContent_MakeAvailable(ccglib)

target_link_libraries(ccglib PRIVATE xtensor)

FetchContent_Declare(
  pcap
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/libpcap
)
FetchContent_MakeAvailable(pcap)

FetchContent_Declare(
  hdf5
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/hdf5
)
FetchContent_MakeAvailable(hdf5)
# Tell HighFive *not* to use find_package(HDF5)
set(HIGHFIVE_FIND_HDF5 OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  HighFive
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/highfive
)
FetchContent_MakeAvailable(HighFive)
target_link_libraries(HighFive INTERFACE hdf5)
target_include_directories(HighFive INTERFACE  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/hdf5/src>  
)
target_include_directories(HighFive INTERFACE  
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/extern/hdf5/config>  
)
# Restore user choice for own project tests
set(BUILD_TESTING ${USER_BUILD_TESTING} CACHE BOOL "Enable testing" FORCE)
message(STATUS "Testing build is ${BUILD_TESTING}")
if(BUILD_TESTING) 
  set(NR_CHANNELS 3)
  set(NR_RECEIVERS 8)
  set(NR_POLARIZATIONS 2)
  set(NR_RECEIVERS_PER_BLOCK 64)
  set(NR_BITS 16)
elseif(BUILD_TARGET STREQUAL "LAMBDA")
  set(NR_CHANNELS 8)
  set(NR_RECEIVERS 32)
  set(NR_POLARIZATIONS 2)
  set(NR_RECEIVERS_PER_BLOCK 32)
  set(NR_BITS 8)
else() 
  set(NR_RECEIVERS 576)
  set(NR_CHANNELS 480)
  set(NR_POLARIZATIONS 2)
  set(NR_RECEIVERS_PER_BLOCK 64)
  set(NR_BITS 16)
endif() 
configure_file(include/spatial/tcc_config.h.in ${PROJECT_SOURCE_DIR}/include/spatial/tcc_config.h @ONLY)
# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
   enable_testing()
  add_subdirectory(tests)
  add_subdirectory(${PROJECT_SOURCE_DIR}/extern/googletest)
endif()

