# @file         - faster_rcnn/CMakeLists.txt
# @author       - wdn (dongnian.wang@outlook.com)
# @brief        - mnist编译文件
# 
# Copyright (c) 2021, wdn. All rights reserved.

cmake_minimum_required (VERSION 3.0)

project ("faster_rcnn")

add_executable (faster_rcnn "main_faster_rcnn.cpp"  "faster_rcnn.cpp" "plugin/plugin_faster_rcnn.h")

# 链接
target_link_libraries(
        faster_rcnn PRIVATE 
        common
        ${CUDA_LIBRARIRS} 
        ${TRT_LIBRARIRS}
        ${OpenCV_LIBS}
        )
