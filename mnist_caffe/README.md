# MNIST Caffe

## 1.Description

继承`TrtBase`类实现`mnist`手写数字算法。

## 2.Running

### 2.1 编译生成可执行文件

```shell
cd TensorRT-Base
mkdir build
cd build
cmake ..
make
```

### 2.2 运行 faster rcnn

```shell
PS E:\TensorRT-Base> .\bin\Release\mnist_caffe.exe
[2021-04-26 21:29:51][  WARN ] : Dynamic size input setting invalid!
[2021-04-26 21:29:51][ DEBUG ] : deserialize engine from mnist_caffe/model/mnist.bin
[2021-04-26 21:29:51][ ERROR ] : INVALID_CONFIG: Deserialize the cuda engine failed.
[2021-04-26 21:29:51][ DEBUG ] : build caffe engine with mnist_caffe/model/mnist.prototxt and mnist_caffe/model/mnist.caffemodel
[2021-04-26 21:29:51][ DEBUG ] : Number of Network layers: 9
[2021-04-26 21:29:51][ DEBUG ] : Number of input: 1
[2021-04-26 21:29:51][ DEBUG ] : Input layer:
[2021-04-26 21:29:51][ DEBUG ] : data : 1x28x28x
[2021-04-26 21:29:51][ DEBUG ] : Number of output: 1
[2021-04-26 21:29:51][ DEBUG ] : Output layer:
[2021-04-26 21:29:51][ DEBUG ] : prob : 10x1x1x
[2021-04-26 21:29:51][ DEBUG ] : FP16 support: 1
[2021-04-26 21:29:51][ DEBUG ] : INT8 support: 1
[2021-04-26 21:29:51][ DEBUG ] : Max batchsize: 1
[2021-04-26 21:29:51][ DEBUG ] : Max workspace size: 16777216
[2021-04-26 21:29:51][ DEBUG ] : Number of DLA core: 0
[2021-04-26 21:29:51][ DEBUG ] : Max DLA batchsize: 268435456
[2021-04-26 21:29:51][ DEBUG ] : Current use DLA core: 0
[2021-04-26 21:29:51][ DEBUG ] : build engine ...
[2021-04-26 21:29:57][ DEBUG ] : build Network done
[2021-04-26 21:29:57][ DEBUG ] : serialize engine to mnist_caffe/model/mnist.bin
[2021-04-26 21:29:57][ DEBUG ] : save engine to mnist_caffe/model/mnist.bin ...
[2021-04-26 21:29:57][ DEBUG ] : create execute context and malloc device memory...
[2021-04-26 21:29:57][ DEBUG ] : init engine...
[2021-04-26 21:29:57][ DEBUG ] : malloc device memory
[2021-04-26 21:29:57][ DEBUG ] : nbBingdings: 2
[2021-04-26 21:29:57][ DEBUG ] : input:
[2021-04-26 21:29:57][ DEBUG ] : binding bindIndex: 0, name: data, size in byte: 3136
[2021-04-26 21:29:57][ DEBUG ] : binding dims with 3 dimemsion
[2021-04-26 21:29:57][ DEBUG ] : output:
[2021-04-26 21:29:57][ DEBUG ] : binding bindIndex: 1, name: prob, size in byte: 40
[2021-04-26 21:29:57][ DEBUG ] : binding dims with 3 dimemsion
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@@@@@@@@@@@@#=.  +*=#@@@@@@@
@@@@@@@@@@@*   :.   -@@@@@@@
@@@@@@@@@@#  :#@@:  +@@@@@@@
@@@@@@@@@*  :@@@*  .@@@@@@@@
@@@@@@@@=  =@@@@.  *@@@@@@@@
@@@@@@@=  -@@@@*  =@@@@@@@@@
@@@@@@@  -@@@%:  -@@@@@@@@@@
@@@@@@%  %%+:    *@@@@@@@@@@
@@@@@@@      ..  @@@@@@@@@@@
@@@@@@@#  .=%%: =@@@@@@@@@@@
@@@@@@@@@@@@@#  +@@@@@@@@@@@
@@@@@@@@@@@@@#  @@@@@@@@@@@@
@@@@@@@@@@@@@@  @@@@@@@@@@@@
@@@@@@@@@@@@@#  @@@@@@@@@@@@
@@@@@@@@@@@@@+  @@@@@@@@@@@@
@@@@@@@@@@@@@%  @@@@@@@@@@@@
@@@@@@@@@@@@@@. #@@@@@@@@@@@
@@@@@@@@@@@@@@* :%@@@@@@@@@@
@@@@@@@@@@@@@@@: -@@@@@@@@@@
@@@@@@@@@@@@@@@@= %@@@@@@@@@
@@@@@@@@@@@@@@@@@@@@@@@@@@@@
[2021-04-26 21:29:57][  INFO ] : Output:
[2021-04-26 21:29:57][  INFO ] : 0: 1.04952e-19
[2021-04-26 21:29:57][  INFO ] : 1: 5.1673e-09
[2021-04-26 21:29:57][  INFO ] : 2: 3.33257e-17
[2021-04-26 21:29:57][  INFO ] : 3: 4.6637e-10
[2021-04-26 21:29:57][  INFO ] : 4: 1.66628e-07
[2021-04-26 21:29:57][  INFO ] : 5: 5.3007e-13
[2021-04-26 21:29:57][  INFO ] : 6: 1.51152e-21
[2021-04-26 21:29:57][  INFO ] : 7: 2.91715e-06
[2021-04-26 21:29:57][  INFO ] : 8: 3.99939e-17
[2021-04-26 21:29:57][  INFO ] : 9: 0.999997
[2021-04-26 21:29:57][  INFO ] :
```

