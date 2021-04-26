# Faster RCNN

## 1.Description

继承`TrtBase`类实现`Faster RCNN`算法。

## 2.Running

### 2.1 下载[faster rcnn model](https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz) dataset。

```shell
wget --no-check-certificate https://dl.dropboxusercontent.com/s/o6ii098bu51d139/faster_rcnn_models.tgz?dl=0 -O faster_rcnn/model/faster-rcnn/faster-rcnn.tgz
```

### 2.2 解压模型文件

此demo使用VGG16的网络，将VGG16权重文件放在`faster_rcnn/model`目录下，修改主函数中model路径即可。

### 2.3 编译生成可执行文件

```shell
cd TensorRT-Base
mkdir build
cd build
cmake ..
make
```

### 2.4 运行 faster rcnn

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

