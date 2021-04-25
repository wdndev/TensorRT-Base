/**
 * @file        - base_trt.cpp
 * @author      - wdn
 * @brief       - tensorrt 推理基类
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include <string>
#include <vector>
#include <iostream>
#include <cassert>
#include <fstream>
#include <memory>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvCaffeParser.h>
#include <NvOnnxParser.h>
#include <NvUffParser.h>
#include <NvInferPlugin.h>

#include "base_trt.h"
#include "helper.h"
#include "plugin_factory.h"
#include "NMS_V3/NMS_V3.h"

/**
 * 默认构造函数
*/
TrtBase::TrtBase()
{
    plugin_param_t tst;
    tst.test = 2222;
    // m_plguin_factory = new PluginFactory(tst);
    m_builder = nvinfer1::createInferBuilder(m_logger);
    m_config = m_builder->createBuilderConfig();
}

/**
 * 析构函数
*/
TrtBase::~TrtBase()
{
    if(m_plguin_factory != nullptr)
    {
        delete m_plguin_factory;
        m_plguin_factory = nullptr;
    }
    
    if(m_context != nullptr)
    {
        m_context->destroy();
        m_context = nullptr;
    }

    if(m_engine != nullptr)
    {
        m_engine->destroy();
        m_engine = nullptr;
    }

    if(m_config != nullptr)
    {
        m_config->destroy();
        m_config = nullptr;
    }

    for(size_t i = 0; i < m_binding.size(); i++)
    {
        //cudaFree(m_binding[i]);
        helper::safe_cuda_free(m_binding[i]);
    }
}

/**
 * 从caffe模型创建engine
*/
trt_ret_t TrtBase::create_engine(const std::string& prototxt, const std::string& caffe_model, 
        const std::string& engine_file,const std::vector<std::string>& output_name, int batch_size, infer_precision_t precision)
{
    trt_ret_t ret = TRT_SUCCESS;
    m_batch_size = batch_size;
    m_precision = precision;

    ret = deserialize_engine(engine_file);
    if(ret != TRT_SUCCESS)
    {
        ret = build_caffe_engine(prototxt, caffe_model, engine_file, output_name);
        if(ret != TRT_SUCCESS)
        {
            helper::err << "Could not build engine!" << std::endl;
            return ret;
        }
    }

    helper::debug << "create execute context and malloc device memory..." << std::endl;
    ret = init_engine();

    return ret;
}

/**
 * 从onnx模型创建engine
*/
trt_ret_t TrtBase::create_engine(const std::string& onnx_model, const std::string& engine_file,
        const std::vector<std::string>& output_name, int batch_size, infer_precision_t precision)
{
    trt_ret_t ret = TRT_SUCCESS;
    m_batch_size = batch_size;
    m_precision = precision;

    ret = deserialize_engine(engine_file);
    if(ret != TRT_SUCCESS)
    {
        ret = build_onnx_engine(onnx_model, engine_file, output_name);
        if(ret != TRT_SUCCESS)
        {
            helper::err << "Could not build engine!" << std::endl;
            return ret;
        }
    }

    helper::debug << "create execute context and malloc device memory..." << std::endl;
    ret = init_engine();

    return ret;
}

/**
 * 反序列化文件创建engine
*/
trt_ret_t TrtBase::create_engine_from_deserialize(const std::string& engine_file)
{
    trt_ret_t ret = TRT_SUCCESS;
    ret = deserialize_engine(engine_file);
    if(ret != TRT_SUCCESS)
    {
        helper::err << "Cound NOT deserialize engine!!!" << std::endl;
        return ret;
    }

    helper::info << "deserialize engine deon!" << std::endl;
    helper::debug << "create execute context and malloc device memory..." << std::endl;
    ret = init_engine();
    return ret;
}

/**
 * 同步推理
*/
trt_ret_t TrtBase::inference()
{
    if(m_context == nullptr)
    {
        helper::err << "context is empty, infer error! " << std::endl;
        return TRT_ERR_CONTEXT;
    }

    if(m_flags == 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
    {
        m_context->executeV2(&m_binding[0]);
    }
    else
    {
        m_context->execute(m_batch_size, &m_binding[0]);
    }

    return TRT_SUCCESS;
}

/**
 * 异步推理
*/
trt_ret_t TrtBase::inference_async(const cudaStream_t& stream)
{
    if(m_context == nullptr)
    {
        helper::err << "context is empty, infer error! " << std::endl;
        return TRT_ERR_CONTEXT;
    }

    if(m_flags == 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH))
    {
        m_context->enqueueV2(&m_binding[0], stream, nullptr);
    }
    else
    {
        m_context->enqueue(m_batch_size, &m_binding[0], stream, nullptr);
    }

    return TRT_SUCCESS;
}

/**
 * host->device
*/
trt_ret_t TrtBase::mem_host_to_device(const std::vector<float>& input_data, 
    int bind_index)
{
    if(input_data.size()*sizeof(float) > m_binding_size[bind_index])
    {
        helper::err << "input data out the allocated memory" << std::endl;
        return TRT_ERR_MEM_CPY;
    }

    cudaError_t cuda_ret = cudaMemcpy(m_binding[bind_index], input_data.data(),
                                      m_binding_size[bind_index], cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess)
    {
        helper::err << "host->device copy mem failed!" << std::endl;
        return TRT_ERR_MEM_CPY;
    }

    return TRT_SUCCESS; 
}

/**
 * host->device
*/
trt_ret_t TrtBase::mem_host_to_device(const std::vector<float>& input_data, 
    int bind_index, const cudaStream_t& stream)
{
    if(input_data.size()*sizeof(float) > m_binding_size[bind_index])
    {
        helper::err << "input data out the allocated memory" << std::endl;
        return TRT_ERR_MEM_CPY;
    }

    cudaError_t cuda_ret = cudaMemcpyAsync(m_binding[bind_index], input_data.data(),
                            m_binding_size[bind_index], cudaMemcpyHostToDevice, stream);
    if(cuda_ret != cudaSuccess)
    {
        helper::err << "host->device copy mem failed!" << std::endl;
        return TRT_ERR_MEM_CPY;
    }

    return TRT_SUCCESS; 
}

/**
 * device->host
*/
trt_ret_t TrtBase::mem_device_to_host(std::vector<float>& output_data, 
    int bind_index)
{
    output_data.resize(m_binding_size[bind_index]/sizeof(float));

    cudaError_t cuda_ret = cudaMemcpy(output_data.data(), m_binding[bind_index],
                                      m_binding_size[bind_index], cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess)
    {
        helper::err << "device->host copy mem failed!" << std::endl;
        return TRT_ERR_MEM_CPY;
    }

    return TRT_SUCCESS; 
}

/**
 * device->host
*/
trt_ret_t TrtBase::mem_device_to_host(std::vector<float>& output_data, 
    int bind_index, const cudaStream_t& stream)
{
    output_data.resize(m_binding_size[bind_index]/sizeof(float));

    cudaError_t cuda_ret = cudaMemcpyAsync(output_data.data(), m_binding[bind_index],
                            m_binding_size[bind_index], cudaMemcpyDeviceToHost, stream);
    if(cuda_ret != cudaSuccess)
    {
        helper::err << "device->host copy mem failed!" << std::endl;
        return TRT_ERR_MEM_CPY;
    }

    return TRT_SUCCESS; 
}

/**
 * 设置device
*/
void TrtBase::set_device(int device)
{
    helper::warn << "warning: make sure save engine file match choosed device" << std::endl;
    CUDA_CHECK(cudaSetDevice(device));
}

/**
 * 获取device
*/
int TrtBase::get_device() const
{
    int device = -1;
    CUDA_CHECK(cudaGetDevice(&device));
    if(device != -1)
    {
        return device;
    }
    else
    {
        helper::err << "Get Device Error" << std::endl;
        return -1;
    }   
}

trt_ret_t TrtBase::set_int8_calibrator(const std::string& calibrator_type, 
            const std::vector<std::vector<float>> calibrator_data)
{
    m_precision = INFER_INT8;
    nvinfer1::IInt8Calibrator* calibrator = helper::get_int8_calibrator(
            calibrator_type, m_batch_size, calibrator_data, "calibrator", false);
    helper::info << "Set INT8 inference mode" << std::endl;
    if(!m_builder->platformHasFastInt8())
    {
        helper::warn << "Current platform doesn't support INT8 inference!" << std::endl;
    }
    // m_builder->setInt8Mode(true);
    m_config->setFlag(nvinfer1::BuilderFlag::kINT8);
    m_config->setInt8Calibrator(calibrator);

    return TRT_SUCCESS;
}

/**
 * 动态调整输入大小
*/
trt_ret_t TrtBase::add_dynamic_shape_profile(int batch_size, const std::string& input_name,
    const std::vector<int>& min_dims_vec, const std::vector<int>& opt_dims_vec, const std::vector<int>& max_dims_vec)
{
    const nvinfer1::Dims4& min_dims{batch_size, min_dims_vec[0], min_dims_vec[1], min_dims_vec[2]};
    const nvinfer1::Dims4& opt_dims{batch_size, opt_dims_vec[0], opt_dims_vec[1], opt_dims_vec[2]};
    const nvinfer1::Dims4& max_dims{batch_size, max_dims_vec[0], max_dims_vec[1], max_dims_vec[2]};
    nvinfer1::IOptimizationProfile* profile = m_builder->createOptimizationProfile();
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMIN, min_dims);
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_name.c_str(), nvinfer1::OptProfileSelector::kMAX, max_dims);
    if(!profile->isValid())
    {
        helper::warn << "Dynamic size input setting invalid!" << std::endl;
        return TRT_ERR_DYNAMIC_INPUT;
    }
    m_config->addOptimizationProfile(profile);

    return TRT_SUCCESS;
}


/**
 * 设置binding dimension
*/
void TrtBase::set_binding_dimentsions(std::vector<int>& input_dims, int bind_index)
{
    const nvinfer1::Dims3& dims{input_dims[0],input_dims[1],input_dims[2]};
    m_context->setBindingDimensions(bind_index, dims);
}

/**
 * 获取最大batch size
*/
int TrtBase::get_max_batch_size() const
{
    return m_batch_size;
}

/**
 * 获取设备中的绑定数据指针。
*/
void* TrtBase::get_binding_ptr(int bind_index) const
{
    return m_binding[bind_index];
}

/**
 * 获取绑定数据的大小
*/
size_t TrtBase::get_binding_size(int bind_index) const
{
    return m_binding_size[bind_index];
}

/**
 * 获取绑定数据维度
*/
nvinfer1::Dims TrtBase::get_binding_dims(int bind_index) const
{
    return m_binding_dims[bind_index];
}

/**
 * 获取绑定数据类型
*/
nvinfer1::DataType TrtBase::get_binding_data_type(int bind_index) const
{
    return m_binding_data_type[bind_index];
}

/**
 * 构建engine
*/
trt_ret_t TrtBase::build_engine()
{
    if(m_precision == 1)
    {
        helper::debug << "setFp16Mode" << std::endl;
        if(!m_builder->platformHasFastFp16())
        {
            helper::warn << "the platform do not has fast for fp16" << std::endl;
        }
        m_config->setFlag(nvinfer1::BuilderFlag::kFP16);
    }
    m_builder->setMaxBatchSize(m_batch_size);
    helper::debug << m_batch_size << std::endl;
    m_config->setMaxWorkspaceSize(16 * (1 << 20));

    helper::debug << "FP16 support: " << m_builder->platformHasFastFp16() << std::endl;
    helper::debug << "INT8 support: " << m_builder->platformHasFastInt8() << std::endl;
    helper::debug << "Max batchsize: " << m_builder->getMaxBatchSize() << std::endl;
    helper::debug << "Max workspace size: " << m_config->getMaxWorkspaceSize() << std::endl;
    helper::debug << "Number of DLA core: " << m_builder->getNbDLACores() << std::endl;
    helper::debug << "Max DLA batchsize: " << m_builder->getMaxDLABatchSize() << std::endl;
    helper::debug << "Current use DLA core: " << m_config->getDLACore() << std::endl;
    helper::debug << "build engine ... " << std::endl;
    m_engine = m_builder->buildEngineWithConfig(*m_network, *m_config);
    if(m_engine == nullptr)
    {
        helper::err << "build engine fail!!" << std::endl;
        return TRT_ERR_ENGINE;
    }

    return TRT_SUCCESS;
}

/**
 * 构建caffe engine
*/
trt_ret_t TrtBase::build_caffe_engine(const std::string& prototxt, const std::string& caffe_model,
                                const std::string& engine_file, const std::vector<std::string>& output_name)
{
    helper::debug << "build caffe engine with " << prototxt << " and " << caffe_model << std::endl;
    if(m_builder == nullptr)
    {
        helper::err << "builder is empty!" << std::endl;
        return TRT_ERR_BUILDER;
    }

    m_network = m_builder->createNetworkV2(m_flags);
    if(m_network == nullptr)
    {
        helper::err << "network is empty!" << std::endl;
        return TRT_ERR_NETWOEK;
    }

    nvcaffeparser1::ICaffeParser* parser = nvcaffeparser1::createCaffeParser();
    if(parser == nullptr)
    {
        helper::err << "parser is empty!" << std::endl;
        return TRT_ERR_PAESER;
    }
    nvinfer1::DataType type = m_precision == 1 ? nvinfer1::DataType::kHALF : nvinfer1::DataType::kFLOAT;
    const nvcaffeparser1::IBlobNameToTensor* blob_name_to_tensor = parser->parse(
            prototxt.c_str(), caffe_model.c_str(), *m_network, nvinfer1::DataType::kFLOAT);
    for(auto& s : output_name)
    {
        m_network->markOutput(*blob_name_to_tensor->find(s.c_str()));
    }
    
    helper::debug << "Number of Network layers: " << m_network->getNbLayers() << std::endl;
    helper::debug << "Number of input: " << m_network->getNbInputs() << std::endl;
    helper::debug << "Input layer: " << std::endl;
    for(int i = 0; i < m_network->getNbInputs(); i++)
    {
        helper::debug << m_network->getInput(i)->getName() << " : ";
        nvinfer1::Dims dims = m_network->getInput(i)->getDimensions();
        for(int j = 0; j < dims.nbDims; j++)
        {
            helper::debug << dims.d[j] << "x";
        }
        helper::debug << "\b" << std::endl;
    }
    helper::debug << "Number of output: " << m_network->getNbOutputs() << std::endl;
    helper::debug << "Output layer: " << std::endl;
    for(int i = 0; i < m_network->getNbOutputs(); i++)
    {
        helper::debug << m_network->getOutput(i)->getName() << " : ";
        nvinfer1::Dims dims = m_network->getOutput(i)->getDimensions();
        for(int j = 0; j < dims.nbDims; j++)
        {
            helper::debug << dims.d[j] << "x";
        }
        helper::debug << "\b" << std::endl;
    }

    build_engine();

    helper::debug << "build Network done" << std::endl;

    helper::debug << "serialize engine to " << engine_file << std::endl;
    serialize_engine(engine_file);

    m_builder->destroy();
    m_network->destroy();
    parser->destroy();

    return TRT_SUCCESS;
}

/**
 * 构建onnx engine
*/
trt_ret_t TrtBase::build_onnx_engine(const std::string& onnx_model, const std::string& engine_file,
                                     const std::vector<std::string>& output_name)
{
    helper::debug << "build onnx engine from " << onnx_model << " ..."<< std::endl;
    if(m_builder == nullptr)
    {
        helper::err << "builder is empty!" << std::endl;
        return TRT_ERR_BUILDER;
    }
    m_flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    m_network = m_builder->createNetworkV2(m_flags);
    // NMSPlugin nms_plugin("NonMaxSuppressionV3");
    //nvinfer1::ITensor *tensor;
    // tensor = m_network->addInput("input0", nvinfer1::DataType::kFLOAT, nvinfer1::Dims4{1, 3, 111, 2222});
    //nvinfer1::ITensor *aInputTensor[] = {tensor};
    //m_network->addPluginV2(aInputTensor, 1, nms_plugin);
    if(m_network == nullptr)
    {
        helper::err << "network is empty!" << std::endl;
        return TRT_ERR_NETWOEK;
    }
    nvonnxparser::IParser* parser = nvonnxparser::createParser(*m_network, m_logger);

    if(parser == nullptr)
    {
        helper::err << "parser is empty!" << std::endl;
        return TRT_ERR_PAESER;
    }
    if(!parser->parseFromFile(onnx_model.c_str(), static_cast<int>(m_logger.get_reportable_severity())))
    {
        helper::err << "could not parse onnx engine" << std::endl;
        return TRT_ERR_PAESER;
    }

    if(output_name.size() > 0)
    {
        helper::debug << "unmark original output..." << std::endl;
        for(int i = 0; i < m_network->getNbOutputs(); i++)
        {
            m_network->unmarkOutput(*(m_network->getOutput(i)));
        }
        helper::debug << "mark custom output..." << std::endl;
        for(int i = 0; i < m_network->getNbLayers(); i++)
        {
            nvinfer1::ITensor* output_tensor = m_network->getLayer(i)->getOutput(0);
            for(size_t j = 0; j < output_name.size(); j++)
            {
                std::string layer_name(output_tensor->getName());
                if(layer_name == output_name[j])
                {
                    // helper::debug << "111111111" << std::endl;
                    m_network->markOutput(*output_tensor);
                    break;
                }
            }
        }
    }

    build_engine();

    helper::debug << "build Network done" << std::endl;

    helper::debug << "serialize engine to " << engine_file << std::endl;
    serialize_engine(engine_file);

    m_builder->destroy();
    m_network->destroy();
    parser->destroy();

    return TRT_SUCCESS;
}

/**
 * 初始化engine
 * 
 * @return @c trt_ret_t
*/
trt_ret_t TrtBase::init_engine()
{
    helper::debug << "init engine..." << std::endl;
    m_context = m_engine->createExecutionContext();
    if(m_context == nullptr)
        return TRT_ERR_CONTEXT;
    
    helper::debug << "malloc device memory" << std::endl;
    int num_bindings = m_engine->getNbBindings();
    helper::debug << "nbBingdings: " << num_bindings << std::endl;
    m_binding.resize(num_bindings);
    m_binding_size.resize(num_bindings);
    m_binding_name.resize(num_bindings);
    m_binding_dims.resize(num_bindings);
    m_binding_data_type.resize(num_bindings);

    for(int i = 0; i < num_bindings; i++)
    {
        nvinfer1::Dims dims = m_engine->getBindingDimensions(i);
        nvinfer1::DataType data_type = m_engine->getBindingDataType(i);
        const char* name= m_engine->getBindingName(i);

        // size_t vol = m_batch_size;
        // int vec_dims = m_engine->getBindingVectorizedDim(i);
        // if(-1 != vec_dims)
        // {
        //     int scalar_per_vec = m_engine->getBindingComponentsPerElement(i);
        //     dims.d[vec_dims] = helper::divUp(dims.d[vec_dims], scalar_per_vec);
        //     vol *= scalar_per_vec;
        // }
        // vol *= helper::volume(dims);
        // helper::debug << "vol: " <<  vol * helper::get_element_size(data_type) << std::endl;

        int64_t total_size = helper::volume(dims) * m_batch_size * helper::get_element_size(data_type);
        m_binding_size[i] = total_size;
        m_binding_name[i] = name;
        m_binding_dims[i] = dims;
        m_binding_data_type[i] = data_type;

        if(m_engine->bindingIsInput(i))
        {
            helper::debug << "input: " << std::endl;
        }  
        else
        {
            helper::debug << "output: "<< std::endl;
        }  
        helper::debug << "binding bindIndex: " << i << ", name: " 
                      << name << ", size in byte: " << total_size<< std::endl;
        helper::debug << "binding dims with " << dims.nbDims << " dimemsion" << std::endl;

        // for(int j = 0; j < dims.nbDims; j++)
        // {

        // }

        m_binding[i] = helper::safe_cuda_malloc(total_size);
        if(m_engine->bindingIsInput(i))
            m_input_size++;
    }

    return TRT_SUCCESS;
}

/**
 * 序列化engine
*/
trt_ret_t TrtBase::serialize_engine(const std::string& file_name)
{
    if(file_name == "")
    {
        helper::warn << "empty engine file name, skip save!" << std::endl;
        return TRT_ERR_FILE_NAME;
    }

    if(m_engine == nullptr)
    {
        helper::err << "engine is empty, save engine failed!" << std::endl;
        return TRT_ERR_SERIALIZE;
    }

    helper::debug << "save engine to " << file_name << " ..." << std::endl;
    nvinfer1::IHostMemory* data = m_engine->serialize();
    std::ofstream file;
    file.open(file_name, std::ios::binary | std::ios::out);
    if(!file.is_open())
    {
        helper::err << "read create engine file " << file_name << " failed! " << std::endl;
        return TRT_ERR_OPEN_FILE;
    }
    file.write((const char*)data->data(), data->size());
    file.close();
    data->destroy();
    return TRT_SUCCESS;
}

/**
 * 反序列化engine
*/
trt_ret_t TrtBase::deserialize_engine(const std::string& engine_file)
{
    if(engine_file == "")
    {
        helper::warn << "empty engine file name, skip save!" << std::endl;
        return TRT_ERR_FILE_NAME;
    }
    
    std::ifstream file(engine_file.c_str(), std::ifstream::binary);
    if(!file.is_open())
    {
        helper::err << "read engine file " << engine_file << " failed! " << std::endl;
        return TRT_ERR_OPEN_FILE;
    }

    helper::debug << "deserialize engine from " << engine_file << std::endl;
    auto const start_pos = file.tellg();
    file.ignore(std::numeric_limits<std::streamsize>::max());
    size_t buffer_count = file.gcount();
    file.seekg(start_pos);
    std::unique_ptr<char[]> engine_buffer(new char[buffer_count]);
    file.read(engine_buffer.get(), buffer_count);
    // initLibNvInferPlugins(&m_logger, "");
    m_runtime = nvinfer1::createInferRuntime(m_logger);
    m_engine = m_runtime->deserializeCudaEngine((void*)engine_buffer.get(), buffer_count, nullptr);
    if(m_engine == nullptr)
    {
        m_runtime->destroy();
        return TRT_ERR_DESERIALIZE;
    }    
    m_batch_size = m_engine->getMaxBatchSize();
    helper::debug << "max batch size of deserialized engine: " << m_engine->getMaxBatchSize() << std::endl;

    m_runtime->destroy();
    return TRT_SUCCESS;
}