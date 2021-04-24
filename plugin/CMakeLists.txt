# @file         - plugin/CMakeLists.txt
# @author       - wdn
# @brief        - plugin编译文件
# 
# @copyright Copyright (c) 2021 wdn

cmake_minimum_required (VERSION 3.0)

project ("trt_plugin")

file(GLOB PLUGIN_SOURCE  
        *.cpp
        demo_plugin/*.cpp
        NMS_V3/*.cpp)
# message("bbbbbbbbbbb: " ${PLUGIN_SOURCE})

add_library(trt_plugin SHARED ${PLUGIN_SOURCE})

# 链接库
target_link_libraries(
        trt_plugin PRIVATE 
        ${CUDA_LIBRARIRS} 
        ${TRT_LIBRARIRS}
        )
