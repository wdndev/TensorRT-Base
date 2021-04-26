# @file         - plugin/CMakeLists.txt
# @author       - wdn (dongnian.wang@outlook.com)
# @brief        - plugin编译文件
# 
# @copyright Copyright (c) 2021, wdn. All rights reserved.

cmake_minimum_required (VERSION 3.0)

project ("trt_plugin")

file(GLOB PLUGIN_SOURCE  
        *.cpp
        demo_plugin/*.cpp)

add_library(trt_plugin STATIC ${PLUGIN_SOURCE})

# 链接库
target_link_libraries(
        trt_plugin PRIVATE 
        ${CUDA_LIBRARIRS} 
        ${TRT_LIBRARIRS}
        )
