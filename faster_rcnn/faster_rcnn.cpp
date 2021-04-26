/**
 * @file        - faster_rcnn.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - faster_rcnn
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include "faster_rcnn.h"
#include <cmath>

trt_ret_t FasterRcnnCaffe::build()
{
    std::vector<int> min_dim_vec{1, 50, 50};
    std::vector<int> opt_dim_vec{1, 28, 28};
    std::vector<int> max_dim_vec{1, 20, 20};
    add_dynamic_shape_profile(m_batch_size,m_input_name[0], min_dim_vec, opt_dim_vec, max_dim_vec);
    initLibNvInferPlugins(&m_logger, "");
    create_engine(m_caffe_prototxt, m_caffe_model, m_engine_file, m_output_name, m_batch_size, m_precision);

    nvinfer1::Dims data_dim = get_binding_dims(m_engine->getBindingIndex(m_input_name[0].c_str()));
    m_input_shape = {data_dim.d[0], data_dim.d[1], data_dim.d[2]};  // CHW
    // helper::info << "111111111:" << data_dim.MAX_DIMS << std::endl; 
    // helper::info << "111111111:" << data_dim.nbDims << std::endl; 
    // helper::info << "111111111:" << data_dim.d[0] << std::endl; 
    // helper::info << "111111111:" << data_dim.d[1] << std::endl; 
    // helper::info << "111111111:" << data_dim.d[2] << std::endl; 

    return TRT_SUCCESS;
}

// trt_ret_t FasterRcnnCaffe::process_input(std::string& pgm_name)
// {
//     int pgm_height = 28;
//     int pgm_width = 28;
//     int pgm_channel = 1;
//     std::vector<uint8_t> file_buffer(pgm_height * pgm_width);
//     m_input_data.resize(pgm_height*pgm_width*pgm_channel);
//     read_pgm(pgm_name, file_buffer.data(), 28, 28);
//     for (int i = 0; i < pgm_height * pgm_width; i++)
//     {
//         std::cout << (" .:-=+*#%@"[file_buffer[i] / 26]) << (((i + 1) % pgm_width) ? "" : "\n");
//         m_input_data[i] = 1.0 - float(file_buffer[i] / 255.0);
//     }
//     return TRT_SUCCESS;
// }

trt_ret_t FasterRcnnCaffe::process_input(std::string& img_name)
{
    helper::warn << "11111111111" << std::endl;
    cv::Mat resize_img;
    cv::Mat image = cv::imread(img_name);
    float pixe_mean[3] {102.9801f, 115.9465f, 122.7717f};  
    cv::resize(image, resize_img, cv::Size(m_input_shape[2], m_input_shape[1]));
    m_infer_img = resize_img;
    int img_height = resize_img.rows;
    int img_width = resize_img.cols;
    int img_channel = resize_img.channels();

    resize_img.convertTo(resize_img, CV_32FC3);
    for(int row = 0; row < img_height; row++)
    {
        for(int col = 0; col < img_width; col++)
        {
            resize_img.at<cv::Vec3f>(row, col)[0] -= pixe_mean[0];
            resize_img.at<cv::Vec3f>(row, col)[1] -= pixe_mean[1];
            resize_img.at<cv::Vec3f>(row, col)[2] -= pixe_mean[2];
        }
    }

    
    std::vector<cv::Mat> input_channels(img_channel);
    cv::split(resize_img, input_channels);
    float* result = new float[img_height * img_width * img_channel];
    std::vector<float> test_data;
    float *data = result;
    for(int i = 0; i < img_channel; i++)
    {
        memcpy(data, input_channels[i].data, img_height*img_width * sizeof(float));
        data += img_height*img_width;
    }

    int input_index0 = m_engine->getBindingIndex(m_input_name[0].c_str());
    std::vector<float> img_data;
    img_data.insert(img_data.begin(), result, result + get_binding_size(input_index0)/ helper::get_element_size(get_binding_data_type(input_index0)));  
    m_input_data.insert(m_input_data.begin(), img_data);
    
    return TRT_SUCCESS;
}

void FasterRcnnCaffe::bbox_transform_clip(std::vector<float> &rois, std::vector<float>& bbox_pred, std::vector<float>& pred_bbox,
                            std::vector<float> &img_info)
{
    for(int i = 0; i < m_batch_size * m_nms_max_out; i++)
    {
        float width = rois[i * 4 + 2] - rois[i * 4] + 1;
        float height = rois[i * 4 + 3] - rois[i * 4 + 1] + 1;
        float ctr_x = rois[i * 4] + 0.5 * width;
        float ctr_y = rois[i * 4 + 1] + 0.5 * height;
        float* img_offset = img_info.data() + i/m_nms_max_out * 3;
        for(int j = 0; j < m_output_class_size; j++)
        {
            float dx = bbox_pred[i * m_output_class_size * 4 + j * 4 + 0];
            float dy = bbox_pred[i * m_output_class_size * 4 + j * 4 + 1];
            float dw = bbox_pred[i * m_output_class_size * 4 + j * 4 + 2];
            float dh = bbox_pred[i * m_output_class_size * 4 + j * 4 + 3];

            float pred_ctr_x = dx * width + ctr_x;
            float pred_ctr_y = dy * height + ctr_y;
            float pred_w = exp(dw) * width;
            float pred_h = exp(dh) * height;

            pred_bbox[i * m_output_class_size * 4 + j * 4 + 0]
                = std::max(std::min(pred_ctr_x - 0.5f * pred_w, img_offset[1] - 1.f), 0.f);
            pred_bbox[i * m_output_class_size * 4 + j * 4 + 1]
                = std::max(std::min(pred_ctr_y - 0.5f * pred_h, img_offset[0] - 1.f), 0.f);
            pred_bbox[i * m_output_class_size * 4 + j * 4 + 2]
                = std::max(std::min(pred_ctr_x + 0.5f * pred_w, img_offset[1] - 1.f), 0.f);
            pred_bbox[i * m_output_class_size * 4 + j * 4 + 3]
                = std::max(std::min(pred_ctr_y + 0.5f * pred_h, img_offset[0] - 1.f), 0.f);      
        }
        
    }
}

std::vector<int> FasterRcnnCaffe::nms(std::vector<std::pair<float, int>> score_index, float* bbox,
                        const int class_num, float nms_threshold)
{
    auto overlap1D = [](float x1min, float x1max, float x2min, float x2max)->float
    {
        if(x1min > x2min)
        {
            std::swap(x1min, x2min);
            std::swap(x1max, x2max);
        }
        return x1max < x2min ? 0 : std::min(x1max, x2max) - x2min;
    };
    auto compute_iou = [&overlap1D](float* bbox1, float* bbox2) -> float {
        float overlapX = overlap1D(bbox1[0], bbox1[2], bbox2[0], bbox2[2]);
        float overlapY = overlap1D(bbox1[1], bbox1[3], bbox2[1], bbox2[3]);
        float area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1]);
        float area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1]);
        float overlap2D = overlapX * overlapY;
        float u = area1 + area2 - overlap2D;
        return u == 0 ? 0 : overlap2D / u;
    };

    std::vector<int> indices;
    for(auto score_index : score_index)
    {
        const int index = score_index.second;
        bool keep = true;
        for(int k = 0; k < indices.size(); k++)
        {
            if(keep)
            {
                int idx = indices[k];
                float iou = compute_iou(&bbox[(index * m_output_class_size + class_num) * 4],
                                        &bbox[(idx * m_output_class_size + class_num) * 4]);
                keep = (iou < nms_threshold);
            }
            else
            {
                break;
            }  
        }
        if (keep) 
            indices.push_back(index);
    }
    return indices;
}

trt_ret_t FasterRcnnCaffe::infer()
{
    // std::vector<int> input_dims{pgm_channel, pgm_width, pgm_height};
    // set_binding_dimentsions(input_dims, 0);
    // mem_host_to_device(m_input_data, 0);
    // inference();
    // mem_device_to_host(m_output_data, 1);
    helper::warn << "bbbbbbbbbbbb: " << m_engine->getNbBindings() << std::endl;
    int input_index0 = m_engine->getBindingIndex(m_input_name[0].c_str());
    int input_index1 = m_engine->getBindingIndex(m_input_name[1].c_str());
    int output_index0 = m_engine->getBindingIndex(m_output_name[0].c_str());
    int output_index1 = m_engine->getBindingIndex(m_output_name[1].c_str());
    int output_index2 = m_engine->getBindingIndex(m_output_name[2].c_str());

    //const int data_size = get_binding_size(input_index0)/ helper::get_element_size(get_binding_data_type(input_index1));
    //m_input_data[1].resize(get_binding_size(input_index1)/ helper::get_element_size(get_binding_data_type(input_index1)));
    std::vector<float> img_info_vec = {(float)m_input_shape[1], (float)m_input_shape[2], 1};
    m_input_data.insert(m_input_data.begin() + 1, img_info_vec);
    helper::warn << "2222222222222: " << m_input_data[0].size() << std::endl;
    helper::warn << "2222222222222: " << m_input_data[1].size() << std::endl;
    helper::warn << "2222222222222: " << get_binding_size(input_index0) << std::endl;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    mem_host_to_device(m_input_data[0], input_index0, stream);
    mem_host_to_device(m_input_data[1], input_index1, stream);
    inference_async(stream);
    std::vector<float> output_bbox_pred;
    std::vector<float> output_cls_prob;
    std::vector<float> output_rois;
    mem_device_to_host(output_bbox_pred, output_index0, stream);
    mem_device_to_host(output_cls_prob, output_index1, stream);
    mem_device_to_host(output_rois, output_index2, stream);
    helper::warn << "output_bbox_pred: " << output_bbox_pred.size() << std::endl;
    helper::warn << "output_cls_prob: " << output_cls_prob.size() << std::endl;
    helper::warn << "output_rois: " << output_rois.size() << std::endl;
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);

    for(int i = 0; i < m_batch_size; i++)
    {
        for(int j = 0; j < m_nms_max_out * 4; j++)
        {
            output_rois[i * m_nms_max_out * 4 + j] /= img_info_vec[i * 3 + 2];
        }
    }

    std::vector<float> pred_bbox(output_bbox_pred.size());
    
    bbox_transform_clip(output_rois, output_bbox_pred, pred_bbox, img_info_vec);
    
    const float nms_threshold = 0.3f;
    const float score_threshold = 0.5f;
    for(int i = 0; i < m_batch_size; i++)
    {
        float* bbox = pred_bbox.data() + i * m_nms_max_out * m_output_class_size * 4;
        float* scores = output_cls_prob.data() + i * m_nms_max_out * m_output_class_size;
        for(int cls = 1; cls < m_output_class_size; cls++)
        {
            std::vector<std::pair<float, int>> score_index;
            for(int idx = 0; idx < m_nms_max_out; idx++)
            {
                if(scores[idx * m_output_class_size + cls] > score_threshold)
                {
                    score_index.push_back(std::make_pair(scores[idx * m_output_class_size + cls], idx));
                    std::stable_sort(score_index.begin(), score_index.end(),
                                [](const std::pair<float, int>& pair1, const std::pair<float, int>& pair2){
                                    return pair1.first > pair2.first;
                                });
                }
            }
            std::vector<int> indices = nms(score_index, bbox, cls, nms_threshold);

            helper::info << "indices size is: " << indices.size() << " " << m_output_class[cls] << std::endl;
            // 
            std::string class_name = m_output_class[cls];
            for(int k = 0; k < indices.size(); k++)
            {
                int index = indices[k];
                cv::rectangle(m_infer_img, cv::Point(bbox[index * m_output_class_size * 4 + cls * 4],
                    bbox[index * m_output_class_size * 4 + cls * 4 + 1]), 
                    cv::Point(bbox[index * m_output_class_size * 4 + cls * 4 + 2],
                    bbox[index * m_output_class_size * 4 + cls * 4 + 3]), cv::Scalar(255,0,0), 1);

                cv::putText(m_infer_img, class_name, cv::Point(50,80), cv::FONT_HERSHEY_COMPLEX, 2, 1, 8, 0);
            }
        }
    }
    cv::imwrite("faster_rcnn/result/test.jpg", m_infer_img);

    // cv::imshow("aaa", m_infer_img);
    // cv::waitKey(0);

    return TRT_SUCCESS;
}

trt_ret_t FasterRcnnCaffe::read_pgm(const std::string& file_name, uint8_t* buffer, int in_h, int in_w)
{
    std::ifstream infile(file_name, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), in_h * in_w);
    return TRT_SUCCESS;
}