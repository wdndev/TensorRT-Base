/**
 * @file        - helper.h
 * @author      - wdn
 * @brief       - common
 * 
 * @copyright Copyright (c) 2021 wdn
 * 
 */

#include "helper.h"

#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>

Logger helper::log_base{Logger::Severity::kVERBOSE};
LogStreamConsumer helper::debug{LOG_DEBUG(helper::log_base)};
LogStreamConsumer helper::info{LOG_INFO(helper::log_base)};
LogStreamConsumer helper::warn{LOG_WARN(helper::log_base)};
LogStreamConsumer helper::err{LOG_ERROR(helper::log_base)};
LogStreamConsumer helper::fatal{LOG_FATAL(helper::log_base)};

void helper::set_reportable_severity(Logger::Severity severity)
{
    log_base.set_reportable_severity(severity);
    info.set_reportable_severity(severity);
    debug.set_reportable_severity(severity);
    warn.set_reportable_severity(severity);
    err.set_reportable_severity(severity);
    fatal.set_reportable_severity(severity);
}

const std::string helper::get_error_string(trt_ret_t status)
{
     switch (status)
    {
        case TRT_SUCCESS:                       
            return "Error flag: " + std::to_string(status) + ", Error string: Running Succeed!";
        // case VINO_SEG_INPUT_NUM_ERROR:          
        //     return "Error flag: " + std::to_string(status) + ", Error string: Model Supports topologies only with 1 input!!!";
        // case VINO_SEG_INPUT_DIM_ERROR:          
        //     return "Error flag: " + std::to_string(status) + ", Error string: 3-channel 4-dimensional model's input is expected!!!";
        // case VINO_SEG_OUTPUT_NUM_ERROR:         
        //     return "Error flag: " + std::to_string(status) + ", Error string: Model supports topologies only with 1 output!!!";
        // case VINO_SEG_OUTPUT_DIM_ERROR:         
        //     return "Error flag: " + std::to_string(status) + ", Error string: Unexpected output blob shape. Only 4D and 3D output blobs are supported.!!!";
        // case VINO_SPLIT_STRING_ERROR:           
        //     return "Error flag: " + std::to_string(status) + ", Error string: Split String Error!!!";
        // case VINO_PARSE_YOLO_ERROR:             
        //     return "Error flag: " + std::to_string(status) + ", Error string: Parse YOLO outputs Error!!!";
    };

    return "Unkonw status code " + std::to_string(status); 

}

int64_t helper::volume(const nvinfer1::Dims& d)
{
    return std::accumulate(d.d, d.d + d.nbDims, 1, std::multiplies<int64_t>());
}

unsigned int helper::get_element_size(nvinfer1::DataType t)
{
    switch (t)
    {
        case nvinfer1::DataType::kINT32: return 4;
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF: return 2;
        case nvinfer1::DataType::kBOOL:
        case nvinfer1::DataType::kINT8: return 1;
        default: throw std::runtime_error("Invalid DataType.");
    }
}

/**
 * 安全申请显存
*/
void* helper::safe_cuda_malloc(size_t mem_size)
{
    void* device_mem;
    // std::cout << "aafffffff: " << mem_size << std::endl;
    CUDA_CHECK(cudaMalloc(&device_mem, mem_size));
    if(device_mem == nullptr)
    {
        helper::err << "Out of memory" << std::endl;
        exit(1);
    }

    return device_mem;
}

/**
 * 安全释放显存
*/
void helper::safe_cuda_free(void* device_mem)
{
    CUDA_CHECK(cudaFree(device_mem));
}

void helper::enable_DLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, int useDLACore, bool allow_GPU_fallback)
{
    if (useDLACore >= 0)
    {
        if (builder->getNbDLACores() == 0)
        {
            std::cerr << "Trying to use DLA core " << useDLACore << " on a platform that doesn't have any DLA cores"
                      << std::endl;
            assert("Error: use DLA core on a platfrom that doesn't have any DLA cores" && false);
        }
        if (allow_GPU_fallback)
        {
            config->setFlag(nvinfer1::BuilderFlag::kGPU_FALLBACK);
        }
        if (!builder->getInt8Mode() && !config->getFlag(nvinfer1::BuilderFlag::kINT8))
        {
            // User has not requested INT8 Mode.
            // By default run in FP16 mode. FP32 mode is not permitted.
            builder->setFp16Mode(true);
            config->setFlag(nvinfer1::BuilderFlag::kFP16);
        }
        config->setDefaultDeviceType(nvinfer1::DeviceType::kDLA);
        config->setDLACore(useDLACore);
        config->setFlag(nvinfer1::BuilderFlag::kSTRICT_TYPES);
    }
}

