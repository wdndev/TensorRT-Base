/**
 * @file        - mnist_onnx.h
 * @author      - wdn
 * @brief       - mnist_onnx
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */

#ifndef MNIST_ONNX
#define MNIST_ONNX

#include "helper.h"
#include "timer.h"
#include "base_trt.h"

class MnistOnnx : public TrtBase
{
public:
    MnistOnnx(const std::string& onnx_model, const std::string& engine_file, int batch_size, 
        const std::vector<std::string>& output_name, infer_precision_t precision) 
        : TrtBase(), m_onnx_model(onnx_model),m_batch_size(batch_size), 
        m_output_name(output_name), m_precision(precision), m_engine_file(engine_file) {}

    trt_ret_t build();

    trt_ret_t process_input(std::string& pgm_name);

    trt_ret_t infer();

    trt_ret_t process_result();

private:
    std::string m_onnx_model;
    std::string m_engine_file;
    int m_batch_size;
    std::vector<std::string> m_output_name;
    infer_precision_t m_precision;

    std::vector<float> m_input_data;
    std::vector<float> m_output_data;

    int m_pgm_height;
    int m_pgm_width;
    int m_pgm_channel;

    trt_ret_t read_pgm(const std::string& file_name, uint8_t* buffer, int in_h, int in_w);
};

#endif // MNIST_ONNX