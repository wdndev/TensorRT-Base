/**
 * @file        - int8_calibrator.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - TensorRT INT8量化类实现
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include <iterator>
#include <algorithm>
#include <fstream>
#include <memory>
#include <string.h>

#include "int8_calibrator.h"

nvinfer1::IInt8Calibrator* helper::get_int8_calibrator(
                const std::string& calibrator_type,
                int batch_size, 
                const std::vector<std::vector<float>>& data,
                const std::string& calib_data_name, 
                bool read_cache)
{
    if(calibrator_type == "Int8EntropyCalibrator2")
    {
        return new Int8EntropyCalibrator2(batch_size, data, calib_data_name, read_cache);
    }
    else
    {
        helper::debug << "[INT8] : Unsupport calibrator type "<< std::endl;
        return nullptr;
    }
    
}


Int8EntropyCalibrator2::Int8EntropyCalibrator2(int batch_size, 
            const std::vector<std::vector<float>>& data,
            const std::string& calib_data_name , 
            bool read_cache)
            : m_calib_data_name(calib_data_name), 
            m_batch_size(batch_size),
            m_read_cache(read_cache)
{
    m_data.reserve(data.size());
    m_data = data;

    m_input_count = batch_size * data[0].size();
    m_current_batch_data = new float(m_input_count);
    m_current_batch_idx = 0;
    m_device_input = helper::safe_cuda_malloc(m_input_count * sizeof(float));
}

Int8EntropyCalibrator2::~Int8EntropyCalibrator2()
{
    helper::safe_cuda_free(m_device_input);
    if(m_current_batch_data)
        delete[] m_current_batch_data;
}

int Int8EntropyCalibrator2::getBatchSize() const
{
    helper::debug << "[INT8] : get batch size : " << m_batch_size << std::endl;
    return m_batch_size;
}

bool Int8EntropyCalibrator2::getBatch(void* bindings[], const char* names[], int nb_bindings)
{
    helper::debug << "[INT8] : name: " << names[0] << ", nbBindings: " << nb_bindings << std::endl;

    if(m_current_batch_idx + m_batch_size > int(m_data.size()))
    {
        return false;
    }

    float* ptr = m_current_batch_data;
    size_t img_size = m_input_count / m_batch_size;
    auto iter = m_data.begin() + m_current_batch_idx;

    std::for_each(iter, iter + m_batch_size, [=, &ptr](std::vector<float>& val)
    {
        assert(img_size == val.size());
        memcpy(ptr, val.data(), img_size*sizeof(float));

        ptr += img_size;
    });

    CUDA_CHECK(cudaMemcpy(m_device_input, m_current_batch_data, 
            m_input_count*sizeof(float), cudaMemcpyHostToDevice));
    
    bindings[0] = m_device_input;
    helper::debug << "[INT8] : load batch " << m_current_batch_idx << " to "
                  << m_current_batch_idx + m_batch_size - 1 << std::endl;
    m_current_batch_idx += m_batch_size;

    return true;
}

const void* Int8EntropyCalibrator2::readCalibrationCache(std::size_t& length)
{
    m_calibration_cache.clear();
    std::ifstream input(m_calib_data_name + ".calib", std::ios::binary);
    input >> std::noskipws;
    if(m_read_cache && input.good())
    {
        std::copy(std::istream_iterator<char>(input), 
            std::istream_iterator<char>(), std::back_inserter(m_calibration_cache));
        
        length = m_calibration_cache.size();
        return length ? &m_calibration_cache[0] : nullptr;
    }
}

void Int8EntropyCalibrator2::writeCalibrationCache(const void* cache, std::size_t length)
{
    std::ofstream output(m_calib_data_name + ".calib", std::ios::binary);
    output.write(reinterpret_cast<const char*>(cache), length);
}