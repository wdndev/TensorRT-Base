/**
 * @file        - demo_plugin.h
 * @author      - wdn (you@domain.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef TRT_DEMO_PLUGIN_H
#define TRT_DEMO_PLUGIN_H

#include <string>
#include <vector>

#include <NvInferPlugin.h>
#include <NvInferRuntimeCommon.h>

#include "helper.h"

namespace nvinfer1
{
namespace plugin
{
class DemoPlugin : public nvinfer1::IPluginV2Ext
{
public:
    /**
     * parse 阶段构造函数
    */
    DemoPlugin(const std::string layer_name);

    /**
     * deserialize 阶段构造函数
    */
    DemoPlugin(const std::string layer_name, const void* data, size_t length);

    /**
     * 注意：删掉默认构造函数
    */
    DemoPlugin() = delete;

    const char* getPluginType() const override;

    const char* getPluginVersion() const override;

    /**
     * 返回op返回多少个Tensor
    */
    int getNbOutputs() const override;

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override;

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override;
    // void configureWithFormat(const nvinfer1::Dims* inputDims, int nbInputs, 
    //         const nvinfer1::Dims* outputDims, int nbOutputs, nvinfer1::DataType type, 
    //         nvinfer1::PluginFormat format, int maxBatchSize) override;

    /**
     * 初始化函数
    */
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

class DemoPluginCreator : public nvinfer1::IPluginCreator
{
public:
    DemoPluginCreator();

    const char* getPluginName() const override;

    const char* getPluginVersion() const override;

    const nvinfer1::PluginFieldCollection* getFieldNames() override;

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override;

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override;

    void setPluginNamespace(const char* pluginNamespace) override;

    const char* getPluginNamespace() const override;
private:
    nvinfer1::PluginFieldCollection m_fc;
    std::string m_namespace;
    std::string m_plugin_version;
    std::string m_plugin_name;
};  // DemoPluginCreator

} // namespace plugin
} // namespace nvinfer1
#endif // TRT_DEMO_PLUGIN_H
