# Common

## 1.Description

TensorRT Base common库，包括TensorRT解析ONNX、Caffe模型封装，日志模块，CPU/GPU计时，INT8量化和返回值处理。

## 2.file

```
  ├── base_trt.cpp: TensorRT 推理类实现
  ├── base_trt.h: TensorRT 推理类定义
  ├── CMakeLists.txt: 生成trt动态库CMake配置文件
  ├── helper.cpp: 公共函数实现
  ├── helper.h: 一些公共函数定义
  ├── int8_calibrator.cpp: TensorRT INT8量化类
  ├── int8_calibrator.h: TensorRT INT8量化类
  ├── logger.h: 日志类
  ├── timer.h: CPU/GPU计时类
  └── trt_return_type.h: TensorRT Base返回值
```

## 3.Logger Usage

在helper公共函数文件中，已定义logger的输出，分别有5个等级，从低到高依次为：`debug`, `info`, `warn`, `err`, `fatal`。

```C++
#include "helper.h"

// debug信息输出，一般用于调试
helper::debug << "hello world！" << msg << std::endl;

// info信息输出，一般用于输出某些提示信息，供用户参考
helper::info << "hello world！" << msg << std::endl;

// warn信息输出，警告信息，提示错误，但此时程序可以运行
helper::warn << "hello world！" << msg << std::endl;

// err信息输出，错误信息，提示这样操作违规，不可操作，程序不能运行
helper::err << "hello world！" << msg << std::endl;

// fatal信息输出，内部错误信息
helper::debug << "hello world！" << msg << std::endl;
```

## 4.Timer Usage

CPU/GPU计时类，封装在helper.h文件中，可直接调用。

CPU计时demo

```C++
#include "helper.h"

helper::PreciseCpuTime cpu_time;
cpu_time.start();
Sleep(1000);
cpu_time.stop();
helper::info << "time: " << cpu_time.milliseconds() << "ms" << std::endl;  // 1000
Sleep(1000);
cpu_time.start();
Sleep(1000);
cpu_time.stop();
helper::info << "time: " << cpu_time.milliseconds() << "ms" << std::endl;  // 1000
cpu_time.continuation();
Sleep(1000);
cpu_time.stop();
helper::info << "time: " << cpu_time.milliseconds() << "ms" << std::endl;  // 2000

delete gpu_time
```

GPU计时demo

```C++
#include "helper.h"

cudaStream_t stream;
helper::PreciseGpuTime gpu_time = helper::PreciseGpuTime(stream);
gpu_time.start();
Sleep(1000);
gpu_time.stop();
helper::info << "gpu time: " << gpu_time.milliseconds() << "ms" << std::endl;
gpu_time.start();
Sleep(1000);
gpu_time.stop();
helper::info << "gpu time: " << gpu_time.milliseconds() << "ms" << std::endl;
gpu_time.continuation();
Sleep(1000);
gpu_time.stop();
helper::info << "gpu time: " << gpu_time.milliseconds() << "ms" << std::endl;

delete gpu_time
```

## 5.TrtBase Usage

支持解析ONNX模型和Caffe模型，可继承修改继续封装或直接实例化使用；

支持INT8量化。

直接实例化`TrtBase`：

```C++
const std::string onnx_model = "mnist_onnx/model/mnist.onnx";
const std::string engine_file = "";
const std::vector<std::string> output_name{"Plus214_Output_0"};
int batch_size = 1;
infer_precision_t precision = INFER_FP32;
TrtBase trt_base = new TrtBase();
trt_base->create_engine(onnx_model, engine_file, output_name, batch_size, precision);
int input_data_size = 300;
std::vector<float> input_data(input_data_size, 0.5);
std::vector<int> input_dims{1,10,30};
trt_base->set_binding_dimentsions(input_dims, 0);
trt_base->mem_host_to_device(input_data, 0);
trt_base->inference();
std::vector<float> output_data;
trt_base->mem_device_to_host(output_data, 1);
// process output data
```

继承`TrtBase`，进一步封装，见 `mnist_onnx` sample。