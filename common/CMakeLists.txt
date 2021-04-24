# @file         - common/CMakeLists.txt
# @author       - wdn
# @brief        - common编译文件
# 
# @copyright Copyright (c) 2021 wdn

cmake_minimum_required (VERSION 3.0)

project ("common")

# aux_source_directory(. COMMON_SRC)
file(GLOB COMMON_SOURCE *.cpp )

# file(GLOB_RECURSE trt_source "helper.h" "helper.cpp" "logging.h" "timer.h")

add_library(common SHARED ${COMMON_SOURCE})

# 链接库
target_link_libraries(
        common PRIVATE 
        trt_plugin
        ${CUDA_LIBRARIRS} 
        ${TRT_LIBRARIRS}
        )
