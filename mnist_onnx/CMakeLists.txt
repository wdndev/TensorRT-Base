# @file mnist_onnx/CMakeLists.txt
# @author wdn
# @brief  mnist编译文件
# 
# @copyright Copyright (c) 2021, wdn. All rights reserved.

cmake_minimum_required (VERSION 3.0)

project ("mnist_onnx")

add_executable (mnist_onnx "main_mnist_onnx.cpp"  "mnist_onnx.cpp")

# 链接库
target_link_libraries(
        mnist_onnx PRIVATE 
        common
        ${CUDA_LIBRARIRS} 
        )
