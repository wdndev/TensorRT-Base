# TensorRT Base

## 1.Description
该项目封装TensorRT加速深度学习模型，支持caffe和onnx格式模型。

主要来自TensorRT C++ sample及部分Github参考代码。

# 2.Environment

推荐使用Ubuntu运行

## 2.1 Ubuntu

* TensorRT 7+
* GUN工具（g++ 7.5.0）
* CMake工具（3.10+）

## 2.2 Windows

* TensorRT 7+
* Visual Studio 2017
* CMake工具（3.10+）

# 3.Directory

```
  ├── bin: 生成可执行文件目录
  ├── build: CMake工具构建目录
  ├── common: TensorRT基础类和一些共用函数，编译为库供调用
  ├── lib: 生成库目录
  ├── plugin: TensorRT plugin目录，用于存放自定义plugin
  ├── CMakeLists.txt: 根目录CMake配置文件
  ├── mnist_caffe: TensorRT解析caffe模型demo
  ├── mnist_onnx: TensorRT解析ONNX模型demo
  └── faster_rcnn: TensoRT加速Faster RCNN算法
```

# 4.Running

修改根目录下CMakeLists.txt文件中`CUDA`和`TensorRT`目录，cudnn默认和CUDA安装在同一目录下。

```cmake
set(CUDA_ROOT_DIR "D:/Software/CUDA11.0/development")
set(TRT_ROOT_DIR "D:/Software/CUDA11.0/development")
```

运行以下命令

```shell
cd TensorRT-Base
mkdir build
cd build
cmake ..
make
```

编译完成后，在bin文件夹中会生成可执行文件，lib目录下会生成库文件，由于Windows环境下配置动态库较麻烦，默认生成静态库，可自己修改。

运行bin文件夹中可执行文件，即可加速深度学习模型，以faster rcnn demo为例

```shell
PS E:\TensorRT-Base> .\bin\Release\faster_rcnn.exe
[2021-04-25 20:48:32][  WARN ] : Dynamic size input setting invalid!
[2021-04-25 20:48:32][ DEBUG ] : deserialize engine from faster_rcnn/model/faster_rcnn.bin
[2021-04-25 20:48:38][ DEBUG ] : max batch size of deserialized engine: 1
[2021-04-25 20:48:38][ DEBUG ] : create execute context and malloc device memory...
[2021-04-25 20:48:38][ DEBUG ] : init engine...
[2021-04-25 20:48:38][ DEBUG ] : malloc device memory
[2021-04-25 20:48:38][ DEBUG ] : nbBingdings: 5
[2021-04-25 20:48:38][ DEBUG ] : input: 
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 0, name: data, size in byte: 2250000
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : input: 
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 1, name: im_info, size in byte: 12  
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : output: 
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 2, name: rois, size in byte: 4800
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : output:
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 3, name: bbox_pred, size in byte: 100800
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 4 dimemsion
[2021-04-25 20:48:38][ DEBUG ] : output:
[2021-04-25 20:48:38][ DEBUG ] : binding bindIndex: 4, name: cls_prob, size in byte: 25200
[2021-04-25 20:48:38][ DEBUG ] : binding dims with 4 dimemsion
[2021-04-25 20:48:38][  WARN ] : output_bbox_pred: 25200
[2021-04-25 20:48:38][  WARN ] : output_cls_prob: 6300
[2021-04-25 20:48:38][  WARN ] : output_rois: 1200
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 aeroplane
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bicycle
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bird
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 boat
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bottle
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 bus
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 car
[2021-04-25 20:48:38][  INFO ] : indices size is: 1 cat
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 chair
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 cow
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 diningtable
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 dog
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 horse
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 motorbike
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 person
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 pottedplant
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 sheep
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 sofa
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 train
[2021-04-25 20:48:38][  INFO ] : indices size is: 0 tvmonitor
```





