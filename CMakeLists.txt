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

  # Let's ensure -std=c++xx instead of -std=g++xx
  set(CMAKE_CXX_EXTENSIONS OFF)
  set(CMAKE_CXX_STANDARD 20)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_EXTENSIONS OFF)
  #set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE "include-what-you-use")
  # Let's nicely support folders in IDEs
  set_property(GLOBAL PROPERTY USE_FOLDERS ON)

  # Save user choice in another variable
  set(USER_BUILD_TESTING ${BUILD_TESTING})
if(CMAKE_BUILD_TYPE MATCHES Debug)
add_library(sanitizer_flags INTERFACE)
    #target_compile_options(sanitizer_flags INTERFACE 
    #  -fsanitize=address -fno-omit-frame-pointer -g)
    #target_link_options(sanitizer_flags INTERFACE -fsanitize=address)
    # add_compile_options(-fsanitize=address -fno-omit-frame-pointer -g)
    #add_link_options(-fsanitize=address)
endif()
  # Disable testing for dependencies forcibly
  set(BUILD_TESTING OFF CACHE BOOL "Disable tests for dependencies" FORCE)
  find_package(CUDAToolkit REQUIRED)
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
  spdlog
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/spdlog
)
FetchContent_MakeAvailable(spdlog)
target_include_directories(spdlog PRIVATE ${spdlog_SOURCE_DIR})

find_package(ZLIB REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
find_package(CFITSIO REQUIRED)

message(STATUS "HDF5_FOUND is ${HDF5_FOUND}")
message(STATUS "HDF5_ROOT is ${HDF5_ROOT}")
message(STATUS "HDF5_INCLUDE_DIRS is ${HDF5_INCLUDE_DIRS}")
message(STATUS "ZLIB_LIBRARIES: ${ZLIB_LIBRARIES}")
message(STATUS "ZLIB_INCLUDE_DIRS: ${ZLIB_INCLUDE_DIRS}")

#include_directories(${ZLIB_INCLUDE_DIRS} ${HDF5_INCLUDE_DIRS})

FetchContent_Declare(
  HighFive
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/extern/highfive
)
FetchContent_MakeAvailable(HighFive)

# Restore user choice for own project tests
set(BUILD_TESTING ${USER_BUILD_TESTING} CACHE BOOL "Enable testing" FORCE)
message(STATUS "Testing build is ${BUILD_TESTING}")


# Silence warnings from external dependencies
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/extern)
# Testing only available if this is the main app
# Emergency override MODERN_CMAKE_BUILD_TESTING provided as well
if((CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME OR MODERN_CMAKE_BUILD_TESTING)
   AND BUILD_TESTING)
   enable_testing()
  add_subdirectory(tests)
  add_subdirectory(${PROJECT_SOURCE_DIR}/extern/googletest)
endif()

