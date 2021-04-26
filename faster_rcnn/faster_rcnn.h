/**
 * @file        - faster_rcnn.h
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - faster_rcnn
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */

#ifndef FASTER_RCNN_H
#define FASTER_RCNN_H

#include <opencv2/opencv.hpp>
#include <map>

#include "helper.h"
#include "timer.h"
#include "base_trt.h"

// struct detection_object_t
// {
//     int xmin, ymin, xmax, ymax;

//     //float 
    
//     detection_object_t(float x, float y, float h, float w)
//     {
//         this->xmin = static_cast<int> (x - w / 2);
//         this->ymin = static_cast<int> (y - h / 2);
//         this->xmax = static_cast<int> (this->xmin + w);
//         this->ymax = static_cast<int> (this->ymin + h);
//     }

//     detection_object_t(float xmin, float ymin, float xmax, float ymax)
//     {
//         this->xmin = static_cast<int> (xmin);
//         this->ymin = static_cast<int> (ymin);
//         this->xmax = static_cast<int> (xmax);
//         this->ymax = static_cast<int> (ymax);
//     }

// };


class FasterRcnnCaffe : public TrtBase
{
public:
    FasterRcnnCaffe(const std::string& caffe_prototxt, const std::string& caffe_model, const std::string& engine_file, 
        int batch_size, const std::vector<std::string>& input_name, const std::vector<std::string>& output_name, infer_precision_t precision) 
        : TrtBase(), m_caffe_prototxt(caffe_prototxt), m_caffe_model(caffe_model), m_batch_size(batch_size), 
        m_input_name(input_name), m_output_name(output_name), m_precision(precision), m_engine_file(engine_file) 
    {
        m_output_class = {"background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car",
        "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
        "train", "tvmonitor"};
    }

    trt_ret_t build();

    trt_ret_t process_input(std::string& img_name);

    trt_ret_t infer();

    trt_ret_t process_result();

    void bbox_transform_clip(std::vector<float> &rois, std::vector<float>& deltas, std::vector<float>& pred_bbox,
                            std::vector<float> &img_info);

    std::vector<int> nms(std::vector<std::pair<float, int>> score_index, float* bbox,
                        const int class_num, float nms_threshold);

private:
    std::string m_caffe_model;
    std::string m_caffe_prototxt;
    std::string m_engine_file;
    int m_batch_size;
    std::vector<std::string> m_output_name;
    std::vector<std::string> m_input_name;
    infer_precision_t m_precision;

    int m_output_class_size{21};
    int m_nms_max_out{300};

    std::vector<std::string> m_output_class;

    // std::map<std::string, std::vector<float>> m_input_data;
    std::vector<std::vector<float>> m_input_data;
    std::vector<std::vector<float>> m_output_data;

    std::vector<int> m_input_shape;

    cv::Mat m_infer_img;

    trt_ret_t read_pgm(const std::string& file_name, uint8_t* buffer, int in_h, int in_w);
};

#endif // FASTER_RCNN_H