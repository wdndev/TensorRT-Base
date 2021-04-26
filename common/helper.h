/**
 * @file        - helper.h
 * @author      - wdn (dongnianwang@outlook.com)
 * @brief       - 一些公共函数定义
 * 
 * Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_HELPER_H
#define TRT_HELPER_H

#include <numeric>
#include <fstream>
#include <map>

#include <cuda_runtime_api.h>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>
#include <NvOnnxParser.h>

#include "logger.h"
#include "timer.h"
#include "trt_return_type.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call_str)                                                                                    \
    {                                                                                                           \
        cudaError_t error_code = call_str;                                                                      \
        if (error_code != cudaSuccess) {                                                                        \
            helper::err << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__ << std::endl;   \
            exit(0);                                                                                            \
        }                                                                                                       \
    }
#endif

namespace helper
{
/**
 * demo:
 * 
 *   helper::PreciseCpuTime cpu_time;
 *   cpu_time.start();
 *   Sleep(1000);
 *   cpu_time.stop();
 *   helper::info << "time: " << cpu_time.milliseconds() << "ms" << std::endl;  // 1000
 *   Sleep(1000);
 *   cpu_time.start();
 *   Sleep(1000);
 *   cpu_time.stop();
 *   helper::info << "time: " << cpu_time.milliseconds() << "ms" << std::endl;  // 1000
 *   cpu_time.continuation();
 *   Sleep(1000);
 *   cpu_time.stop();
 *   helper::info << "time: " << cpu_time.milliseconds() << "ms" << std::endl;  // 2000
*/
using PreciseCpuTime = CpuTimer<std::chrono::high_resolution_clock>;

/**
 * demo:
 * 
 *   cudaStream_t stream;
 *   helper::PreciseGpuTime gpu_time = helper::PreciseGpuTime(stream);
 *   gpu_time.start();
 *   Sleep(1000);
 *   gpu_time.stop();
 *   helper::info << "gpu time: " << gpu_time.milliseconds() << "ms" << std::endl;
 *   gpu_time.start();
 *   Sleep(1000);
 *   gpu_time.stop();
 *   helper::info << "gpu time: " << gpu_time.milliseconds() << "ms" << std::endl;
 *   gpu_time.continuation();
 *   Sleep(1000);
 *   gpu_time.stop();
 *   helper::info << "gpu time: " << gpu_time.milliseconds() << "ms" << std::endl;
*/
using PreciseGpuTime = GpuTimer;

/**
 * 重载 "" _GiB
*/
constexpr long long int operator"" _GiB(long long unsigned int val)
{
    return val * (1 << 30);
}
/**
 * 重载 "" _MiB
*/
constexpr long long int operator"" _MiB(long long unsigned int val)
{
    return val * (1 << 20);
}
/**
 * 重载 "" _KiB
*/
constexpr long long int operator"" _KiB(long long unsigned int val)
{
    return val * (1 << 10);
}

/**
 * 日志输出
*/
extern Logger log_base;
extern LogStreamConsumer debug;
extern LogStreamConsumer info;
extern LogStreamConsumer warn;
extern LogStreamConsumer err;
extern LogStreamConsumer fatal;

/**
 * 设置日志等级
 * 
 * @param severity      - 日志等级
*/
void set_reportable_severity(Logger::Severity severity);

/**
 *  根据错误标志位返回错误字符串
 *
 * @param status            - 错误标志位
 * 
 * @return std::string      - 错误字符串
 * 
 */
const std::string get_error_string(trt_ret_t status);

/**
 * 
*/
int64_t volume(const nvinfer1::Dims& d);

template <typename A, typename B>
inline A divUp(A x, B n)
{
    return (x + n - 1) / n;
}

/**
 * 获取数据类型大小
*/
unsigned int get_element_size(nvinfer1::DataType t);

/**
 * 安全申请显存
*/
void* safe_cuda_malloc(size_t mem_size);

/**
 * 安全释放显存
*/
void safe_cuda_free(void* device_mem);

void enable_DLA(nvinfer1::IBuilder* builder, nvinfer1::IBuilderConfig* config, 
                int useDLACore, bool allow_GPU_fallback = true);

} // namespace helper

#endif // TRT_HELPER_H