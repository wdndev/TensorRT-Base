# Plugin

## 1.Description

TensorRT已经只支持了许多常见的神经网络层,比如卷积, 池化, BN等等. 但是依然还有很多操作和算子是不支持的,所以TensorRT提供了接口让我们可以编写插件来实现自己的自定义层。

随着tensorRT的不断发展(v5->v6->v7)，TensorRT的插件的使用方式也在不断更新。插件接口也在不断地变化，由v5版本的`IPluginV2Ext`，到v6版本的`IPluginV2IOExt`和`IPluginV2DynamicExt`。

| TensorRT版本        | 混合精度 | 动态大小输入 | Requires extended runtime |      |
| ------------------- | -------- | ------------ | ------------------------- | ---- |
| IPluginV2Ext        | 5.1      | Limited      | No                        | No   |
| IPluginV2IOExt      | 6.0.1    | General      | No                        | No   |
| IPluginV2DynamicExt | 6.0.1    | General      | Yes                       | Yes  |

## 2.Plugin Class

需要写两个类：

- `DemoPlugin`，继承`IPluginV2Ext`，是插件类，用于写插件具体的实现
- `DemoPluginCreator`，继承`IPluginCreator`，是插件工厂类，用于根据需求创建该插件

对了，插件类继承`IPluginV2DynamicExt`才可以支持动态尺寸，其他插件类接口例如`IPluginV2IOExt`和前者大部分是相似的。

```cpp

class DemoPlugin : public nvinfer1::IPluginV2Ext

class DemoPluginCreator : public nvinfer1::IPluginCreator
```

## 3.DemoPlugin Class

```C++
class DemoPlugin : public nvinfer1::IPluginV2Ext
{
public:
    DemoPlugin(const std::string layer_name);
    DemoPlugin(const std::string layer_name, const void* data, size_t length);
    DemoPlugin() = delete;
    const char* getPluginType() const override;
    const char* getPluginVersion() const override;
    int getNbOutputs() const override;
    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;
    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;
    int initialize() override;
    void terminate() override;
    size_t getWorkspaceSize(int maxBatchSize) const override;
    int enqueue(int batchSize, const void* const* inputs, void** outputs, 
                void* workspace, cudaStream_t stream) override;  
    size_t getSerializationSize() const override;
    void serialize(void* buffer) const override;
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override;
    bool isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const override;
    bool canBroadcastInputAcrossBatch(int inputIndex) const override;
    void configurePlugin(const nvinfer1::Dims* inputDims, int nbInputs, const nvinfer1::Dims* outputDims,
                                 int nbOutputs, const nvinfer1::DataType* inputTypes, 
                                 const nvinfer1::DataType* outputTypes,
                                 const bool* inputIsBroadcast, const bool* outputIsBroadcast, 
                                 nvinfer1::PluginFormat floatFormat, int maxBatchSize) override;
    void destroy() override;
    nvinfer1::IPluginV2Ext* clone() const override;
    void setPluginNamespace(const char* pluginNamespace) override;
    const char* getPluginNamespace() const override;
private:
    const std::string m_layer_name;
    size_t m_cpoy_size;
    std::string m_namespace;
    std::string m_plugin_version;
    std::string m_plugin_name;
    
}; //DemoPlugin
```

## 4.IPluginV2插件的工作流

### 4.1 parse phase/ parse阶段

在模型的parse阶段会通过`CustomPlugin(const Weights *weights, int nbWeights)`创建模型中每一个自定义层的实例，在这个阶段还会调用到`getNbOutputs()`和`getOutputDimensions()`来获取自定义层的输出个数和维度，这个步骤的目的是为了构建整一个模型的工作流。

如果自定义层的输出个数和维度跟其他层匹配不上，parse就会失败.所以如果你的自定义层在parse阶段就parse失败了，可以先检查一下这两个函数的实现。 这个阶段创建的CustomPlugin实例会在engine构建阶段被析构掉。

### 4.2 build engine phase / engine构建阶段

engine构建阶段会再次通过`CustomPlugin(const Weights *weights, int nbWeights)`创建自定义层的实例.然后调用`supportFormat()`函数来检查自定义层的支持的`Datatype`和`PluginFormat`。

 在build的过程中，会调用`configureWithFormat`，根据设定的类型(见参数)对插件进行配置。调用完这个函数之后，自定义层内部的状态和变量应该被配置好了。在这里也会调用getWorksapceSize()，但是这个函数不怎么重要。

最后会调用initialize()，进行初始化。此时已经准备好所有准备的数据和参数可以进行执行了。构建结束后当调用builder， network或者 engine的destroy()函数时，会调用CustomPlugin的destroy()方法析构掉CustomPlugin对象。

### 4.3 save engine phase / 引擎保存阶段

保存引擎到序列化文件会调用`getSerializationSize()`函数来获取序列化所需要的空间，在保存的过程中会调用serialize()函数将自定义层的相关信息序列化到引擎文件。

### 4.4 engine running phase / 引擎推理阶段

在这个阶段会调用用`enqueue()`进行模型推理

### 4.5  inference with engine file / 使用引擎文件进行推理

在使用引擎文件进行推理的过程中,从序列化文件恢复权重和参数,所以会先调用`DemoPlugins(const void *data, size_t length)`读取自定义层的相关信息，然后调用initialize() 进行初始化。

在推理的过程中调用enqueue()进行推理，推理结束后如果在调用engine的destroy方法的时候会调用terminate()函数，释放 掉initialize()申请的资源。

## 5.IPluginCreator Class

IPluginCreator主要用于将编写好的IPlugin插件注册到Plugin Registry。

```c++
class DemoPluginCreator : public nvinfer1::IPluginCreator
{
public:
    /**
     * 默认构造函数
    */
    DemoPluginCreator();
    /**
     * 获取 plugin 名称
    */
    const char* getPluginName() const override;
    /**
     * 获取 plugin 版本
    */
    const char* getPluginVersion() const override;
    /**
     * 获取 file 名称
    */
    const nvinfer1::PluginFieldCollection* getFieldNames() override;
    /**
     * 创建 plugin
    */
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;
    /**
     * 序列化 plugin
    */
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;
    /**
     * 设置 namespace
    */
    void setPluginNamespace(const char* pluginNamespace) override;
    /**
     * 获取 namespace
    */
    const char* getPluginNamespace() const override;
private:
    nvinfer1::PluginFieldCollection m_fc;
    std::string m_namespace;
    std::string m_plugin_version;
    std::string m_plugin_name;
};  // DemoPluginCreator
```

## 6.REGISTER_TENSORRT_PLUGIN

注册plugin宏定义。

```C++
/**
 * 注册plugin ！！！！！！！！！！！！！！
*/
REGISTER_TENSORRT_PLUGIN(DemoPluginCreator);
```

