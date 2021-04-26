/**
 * @file        - main_faster_rcnn.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - faster_rcnn测试
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */

#include <iostream>

#include "helper.h"
#include "faster_rcnn.h"

int main()
{
    const std::string caffe_prototxt = "faster_rcnn/model/faster_rcnn_test_iplugin.prototxt";
    const std::string caffe_model = "faster_rcnn/model/VGG16_faster_rcnn_final.caffemodel";
    const std::string engine_file = "faster_rcnn/model/faster_rcnn.bin";
    int batch_size = 1;
    const std::vector<std::string> input_name{"data", "im_info"};
    const std::vector<std::string> output_name{"bbox_pred", "cls_prob", "rois"};
    infer_precision_t precision = INFER_FP32;


    FasterRcnnCaffe* faster_rcnn = new FasterRcnnCaffe(caffe_prototxt, caffe_model, engine_file, batch_size, input_name, output_name, precision);

    faster_rcnn->build();
    std::string pgm_name = "faster_rcnn/image/3.jpg";
    faster_rcnn->process_input(pgm_name);

    faster_rcnn->infer();


    // faster_rcnn->process_result();

    delete faster_rcnn;
    return 0;
}