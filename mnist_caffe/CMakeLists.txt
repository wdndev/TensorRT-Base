# @file mnist/CMakeLists.txt
# @author wdn
# @brief  mnist编译文件
# 
# @copyright Copyright (c) 2021, wdn. All rights reserved.

cmake_minimum_required (VERSION 3.0)

project ("mnist_caffe")

add_executable (mnist_caffe "main_mnist_caffe.cpp" "mnist_caffe.cpp")

# 链接库
target_link_libraries(
        mnist_caffe PRIVATE 
        common
        ${CUDA_LIBRARIRS} 
        )
