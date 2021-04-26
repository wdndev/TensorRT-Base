/**
 * @file        - trt_return_type.h
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_RETURN_TYPE_H
#define TRT_RETURN_TYPE_H

#include <string>

/**
 * TensorRT 推理返回值
 */
typedef enum {
    TRT_SUCCESS                 =  0,
    TRT_ERR_FILE_NAME           = -1,
    TRT_ERR_ENGINE              = -2,
    TRT_ERR_OPEN_FILE           = -3,
    TRT_ERR_CONTEXT             = -4,
    TRT_ERR_DESERIALIZE         = -5,
    TRT_ERR_SERIALIZE           = -6,
    TRT_ERR_BUILDER             = -7,
    TRT_ERR_NETWOEK             = -8,
    TRT_ERR_PAESER              = -9,
    TRT_ERR_MEM_CPY             = -10,
    TRT_ERR_DYNAMIC_INPUT       = -11,

} trt_ret_t;

#endif // TRT_RETURN_TYPE_H
