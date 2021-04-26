/**
 * @file        - mnist_caffe.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - mnist_caffe
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include "mnist_caffe.h"

trt_ret_t MnistCaffe::build()
{
    std::string input_name = "loss";
    std::vector<int> min_dim_vec{1, 50, 50};
    std::vector<int> opt_dim_vec{1, 28, 28};
    std::vector<int> max_dim_vec{1, 20, 20};
    add_dynamic_shape_profile(m_batch_size,input_name, min_dim_vec, opt_dim_vec, max_dim_vec);
    create_engine(m_caffe_prototxt, m_caffe_model, m_engine_file, m_output_name, m_batch_size, m_precision);

    return TRT_SUCCESS;
}

trt_ret_t MnistCaffe::process_input(std::string& pgm_name)
{
    m_pgm_height = 28;
    m_pgm_width = 28;
    m_pgm_channel = 1;
    std::vector<uint8_t> file_buffer(m_pgm_height * m_pgm_width);
    m_input_data.resize(m_pgm_height*m_pgm_width*m_pgm_channel);
    read_pgm(pgm_name, file_buffer.data(), 28, 28);
    for (int i = 0; i < m_pgm_height * m_pgm_width; i++)
    {
        std::cout << (" .:-=+*#%@"[file_buffer[i] / 26]) << (((i + 1) % m_pgm_width) ? "" : "\n");
        m_input_data[i] = float(file_buffer[i]);
    }
    return TRT_SUCCESS;
}


trt_ret_t MnistCaffe::infer()
{
    std::vector<int> input_dims{m_pgm_channel, m_pgm_width, m_pgm_height};
    set_binding_dimentsions(input_dims, 0);
    mem_host_to_device(m_input_data, 0);
    inference();
    mem_device_to_host(m_output_data, 1);

    return TRT_SUCCESS;
}

trt_ret_t MnistCaffe::process_result()
{
    helper::info << "Output:" << std::endl;
    float val{0.0f};
    int idx{0};
    const int kDIGITS = 10;

    for (int i = 0; i < kDIGITS; i++)
    {
        helper::info << i << ": " << m_output_data[i] << std::endl;
    }
    helper::info << std::endl;

    return TRT_SUCCESS;
}


trt_ret_t MnistCaffe::read_pgm(const std::string& file_name, uint8_t* buffer, int in_h, int in_w)
{
    std::ifstream infile(file_name, std::ifstream::binary);
    assert(infile.is_open() && "Attempting to read from a file that is not open.");
    std::string magic, h, w, max;
    infile >> magic >> h >> w >> max;
    infile.seekg(1, infile.cur);
    infile.read(reinterpret_cast<char*>(buffer), in_h * in_w);
    return TRT_SUCCESS;
}