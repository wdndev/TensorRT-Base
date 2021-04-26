/**
 * @file        - base_trt.h
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief       - tensorrt 推理基类
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_BASE_TRT_H
#define TRT_BASE_TRT_H

#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <algorithm>

#include <NvInfer.h>

#include "helper.h"
#include "trt_return_type.h"
#include "int8_calibrator.h"

/**
 * 运行精度
*/
typedef enum
{
    INFER_FP32 = 0,
    INFER_FP16 = 1,
    INFER_INT8 = 2, 
}infer_precision_t;

// /**
//  * 
// */
// typedef struct
// {
//     std::string name;
//     int64_t size;
//     void* value; 
// }data_t;


class TrtBase
{
public:
    /**
     * 默认构造函数
    */
    TrtBase();

    /**
     * 析构函数
    */
    virtual ~TrtBase();

    /**
     * 从caffe模型创建engine
     * 
     * @param prototxt          - caffe prototxt
     * @param caffe_model        - caffe模型
     * @param engine_file       - file
     * @param output_name     - 模型输出
     * @param batch_size        - max batch size
     * @param precision         - infer精度
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t create_engine(
        const std::string& prototxt,
        const std::string& caffe_model, 
        const std::string& engine_file,
        const std::vector<std::string>& output_name, 
        int batch_size,
        infer_precision_t precision);

    /**
     * 从onnx模型创建engine
     * 
     * @param onnx_model        - onnx模型
     * @param engine_file       - file
     * @param output_name     - 模型输出
     * @param batch_size        - max batch size
     * @param precision         - infer精度
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t create_engine(
        const std::string& onnx_model, 
        const std::string& engine_file,
        const std::vector<std::string>& output_name, 
        int batch_size,
        infer_precision_t precision);

    /**
     * 反序列化文件创建engine
     * 
     * @param engine_file       - file
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t create_engine_from_deserialize(const std::string& engine_file);

    /**
     * 同步推理
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t inference();

    /**
     * 异步推理
     * 
     * @param stream        - cuda流
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t inference_async(const cudaStream_t& stream);

    /**
     * host->device
     * 
     * @param input_data        - 输入数据（host）
     * @param bind_index        - 
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t mem_host_to_device(const std::vector<float>& input_data, int bind_index);

    /**
     * host->device
     * 
     * @param input_data        - 输入数据（host）
     * @param bind_index        - 
     * @param stream            - CUDA stream
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t mem_host_to_device(const std::vector<float>& input_data, int bind_index, const cudaStream_t& stream);

    /**
     * device->host
     * 
     * @param output_data       - 输出数据（host）
     * @param bind_index        - 
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t mem_device_to_host(std::vector<float>& output_data, int bind_index);

    /**
     * device->host
     * 
     * @param output_data       - 输出数据（host）
     * @param bind_index        - 
     * @param stream            - CUDA stream
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t mem_device_to_host(std::vector<float>& output_data, int bind_index, const cudaStream_t& stream);

    /**
     * 设置device
    */
    void set_device(int device);
    
    /**
     * 获取device
    */
    int get_device() const;


    trt_ret_t set_int8_calibrator(const std::string& calibrator_type, 
            const std::vector<std::vector<float>> calibrator_data);

    /**
     * 动态调整输入大小
     * 
     * @param batch_size            - batch_size
     * @param input_name            - input_name
     * @param min_dims_vec          - 输入shape最小
     * @param opt_dims_vec          - 最优 shape
     * @param max_dims_vec          - 输入 shape 最大
     * 
     * @return @c trt_ret_t
     * 
     * @note example
     * 
     *  std::vector<int> min_dims_vec{3, 100, 100};
     *  std::vector<int> opt_dims_vec{3, 224, 224};
     *  std::vector<int> max_dims_vec{3, 300, 300};
    */
    trt_ret_t add_dynamic_shape_profile(int batch_size, const std::string& input_name,
        const std::vector<int>& min_dims_vec, const std::vector<int>& opt_dims_vec, const std::vector<int>& max_dims_vec);


    /**
     * 设置binding dimension
     * 
     * @param input_dims    - 
     * @param bind_index    -
     *
     * @return @c void
    */
    void set_binding_dimentsions(std::vector<int>& input_dims, int bind_index);

    /**
     * 获取最大batch size
    */
    int get_max_batch_size() const;

    /**
     * 获取设备中的绑定数据指针。
     * 
     * @param bind_index    - 
     * 
     * @return void*        - device指针
    */
   void* get_binding_ptr(int bind_index) const;

    /**
     * 获取绑定数据的大小
     * 
     * @param bind_index    - 
     * 
     * @return @c size_t
    */
    size_t get_binding_size(int bind_index) const;

    /**
     * 获取绑定数据维度
     * 
     * @param bind_index    - 
     * 
     * @return @c nvinfer1::Dims
    */
    nvinfer1::Dims get_binding_dims(int bind_index) const;

    /**
     * 获取绑定数据类型
     * 
     * @param bind_index    -
     * 
     * @param @c nvinfer1::DataType
    */
    nvinfer1::DataType get_binding_data_type(int bind_index) const;
    
    std::vector<std::string> m_binding_name;

protected:
    /**
     * 构建engine
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t build_engine();

    /**
     * 构建caffe engine
     * 
     * @param prototxt          - caffe prototxt
     * @param caffe_model       - caffe 模型
     * @param engine_file       - engine 文件名
     * @param output_name     - 模型输出名称
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t build_caffe_engine(const std::string& prototxt, const std::string& caffe_model,
                                 const std::string& engine_file, const std::vector<std::string>& output_name);
    
    /**
     * 构建onnx engine
     * 
     * @param onnx_model        - onnx 模型
     * @param engine_file       - engine 文件名
     * @param output_name     - 模型输出名称
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t build_onnx_engine(const std::string& onnx_model, const std::string& engine_file,
                                const std::vector<std::string>& output_name);

    /**
     * 初始化engine
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t init_engine();

    /**
     * 序列化engine
     * 
     * @param file_name         - engine 保存文件名
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t serialize_engine(const std::string& file_name);

    /**
     * 反序列化engine
     * 
     * @param engine_file       - engine 文件名
     * 
     * @return @c trt_ret_t
    */
    trt_ret_t deserialize_engine(const std::string& engine_file);

protected:
    /**
     * 日志
    */
    Logger m_logger;

    /**
     * 运行模式
    */
    infer_precision_t m_precision;

    /**
     * batch size
    */
    int m_batch_size;

    nvinfer1::NetworkDefinitionCreationFlags m_flags = 0;

    nvinfer1::IBuilder* m_builder = nullptr;

    nvinfer1::INetworkDefinition* m_network = nullptr;

    nvinfer1::IBuilderConfig* m_config = nullptr;

    nvinfer1::ICudaEngine* m_engine = nullptr;

    nvinfer1::IExecutionContext* m_context = nullptr;

    nvinfer1::IRuntime* m_runtime = nullptr;

    std::vector<void*> m_binding;

    std::vector<size_t> m_binding_size;

    std::vector<nvinfer1::Dims> m_binding_dims;

    std::vector<nvinfer1::DataType> m_binding_data_type;

    int m_input_size = 0;

}; // TrtBase

#endif //TRT_BASE_TRT_H