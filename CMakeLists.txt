# @file CMakeLists.txt
# @author wdn
# @brief  工程编译文件
# 
# @copyright Copyright (c) 2021, wdn. All rights reserved.

cmake_minimum_required(VERSION 3.0)

project("tensorrt_base")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) 
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

if(WIN32)
  set(CUDA_ROOT_DIR "D:/Software/CUDA11.0/development")
  set(TRT_ROOT_DIR "D:/Software/CUDA11.0/development")

  # cuda
  set(CUDA_INCLUDE_DIR ${CUDA_ROOT_DIR}/include)
  set(CUDA_LIB_DIR ${CUDA_ROOT_DIR}/lib/x64)
  file(GLOB CUDA_LIBRARIRS "${CUDA_ROOT_DIR}/lib/x64/*.lib")

  # tensorrt
  set(TRT_INCLUDE_DIR ${TRT_ROOT_DIR}/include)
  set(TRT_LIB_DIR ${TRT_ROOT_DIR}/lib/x64)
  file(GLOB TRT_LIBRARIRS "${TRT_ROOT_DIR}/lib/x64/*.lib")

else()
  

  set(CUDA_ROOT_DIR "/usr/local/cuda")
  set(TRT_ROOT_DIR "/opt/TensorRT-7.2.2.3")

    # cuda
  set(CUDA_INCLUDE_DIR ${CUDA_ROOT_DIR}/include)
  set(CUDA_LIB_DIR ${CUDA_ROOT_DIR}/lib64)
  file(GLOB CUDA_LIBRARIRS "${CUDA_ROOT_DIR}/lib64/*.so")

  # tensorrt
  set(TRT_INCLUDE_DIR ${TRT_ROOT_DIR}/include)
  set(TRT_LIB_DIR ${TRT_ROOT_DIR}/lib)
  file(GLOB TRT_LIBRARIRS "${TRT_ROOT_DIR}/lib/*.so")
  
endif()

if(WIN32)
  # Socket limit is 16K(can be raised arbitrarily)
  add_definitions(-DFD_SETSIZE=16384)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_WINSOCK_DEPRECATED_NO_WARNINGS)
endif()

# opencv
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${CUDA_INCLUDE_DIR} ${TRT_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
include_directories(common plugin)
link_directories(${CUDA_LIB_DIR} ${TRT_LIB_DIR})

# 包含子项目。
add_subdirectory ("common")
add_subdirectory ("plugin")
add_subdirectory ("mnist_caffe")
add_subdirectory ("mnist_onnx")
add_subdirectory ("faster_rcnn")



