/**
 * @file        - demo_plugin.cpp
 * @author      - wdn (dongnian.wang@outlook.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include "demo_plugin.h"
#include "helper.h"

using namespace nvinfer1;
using nvinfer1::plugin::DemoPlugin;
using nvinfer1::plugin::DemoPluginCreator;

// log
#define log(...) {  \
    char str[100];  \
    sprintf(str, __VA_ARGS__);  \
    helper::debug << "(@ _ @) Here ---> call " << "[" << __FILE__    \
                  << __FUNCTION__ << "][Line " << __LINE__ << "]"   \
                  << str << std::endl;                              \
}

DemoPlugin::DemoPlugin(const std::string layer_name) : m_layer_name(layer_name)
{
    log(" run here now! ");
    m_plugin_name = "demo_plugin";
    m_plugin_version = "01";
}

DemoPlugin::DemoPlugin(const std::string layer_name, const void* data, size_t length)
    : m_layer_name(layer_name)
{
    log(" run here now! ");
}

const char* DemoPlugin::getPluginType() const
{
    log(" run here now! ");
    return m_plugin_name.c_str();
}

const char* DemoPlugin::getPluginVersion() const
{
    log(" run here now! ");
    return m_plugin_version.c_str();
}

int DemoPlugin::getNbOutputs() const
{
    log(" run here now! ");
    return 1;
}

nvinfer1::Dims DemoPlugin::getOutputDimensions(int index, 
    const nvinfer1::Dims* inputs, int nbInputDims)
{
    log(" run here now! ");
    return nvinfer1::Dims3(inputs[1].d[1],inputs[1].d[2],inputs[1].d[3]);
}

bool DemoPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const
{
    log(" run here now! ");
    return false;
}

int DemoPlugin::initialize()
{
    log(" run here now! ");
    return 0;
}

void DemoPlugin::terminate()
{
    log(" run here now! ");
}

size_t DemoPlugin::getWorkspaceSize(int maxBatchSize) const
{
    log(" run here now! ");
    return 0;
}

int DemoPlugin::enqueue(int batchSize, const void* const* inputs, 
            void** outputs, void* workspace, cudaStream_t stream)
{
    log(" run here now! ");
    return 0;
}

size_t DemoPlugin::getSerializationSize() const
{
    log(" run here now! ");
    return 0;
}

void DemoPlugin::serialize(void* buffer) const
{
    log(" run here now! ");
}

nvinfer1::DataType DemoPlugin::getOutputDataType(int index, 
    const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    log(" run here now! ");
    return nvinfer1::DataType::kFLOAT;
}

bool DemoPlugin::isOutputBroadcastAcrossBatch(int outputIndex, 
    const bool* inputIsBroadcasted, int nbInputs) const
{
    log(" run here now! ");
    return false;
}

bool DemoPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    log(" run here now! ");
    return false;
}

void DemoPlugin::configurePlugin(const nvinfer1::Dims* inputDims, 
    int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, 
    const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,
    const bool* inputIsBroadcast, const bool* outputIsBroadcast, 
    nvinfer1::PluginFormat floatFormat, int maxBatchSize)
{
    log(" run here now! ");
}

void DemoPlugin::destroy()
{
    log(" run here now! ");
}

nvinfer1::IPluginV2Ext* DemoPlugin::clone() const
{
    log(" run here now! ");
    auto* plugin = new DemoPlugin(m_layer_name);
    plugin->setPluginNamespace(m_namespace.c_str());
    return plugin;
}

void DemoPlugin::setPluginNamespace(const char* pluginNamespace)
{
    log(" run here now! ");
}

const char* DemoPlugin::getPluginNamespace() const
{
    log(" run here now! ");
    return m_namespace.c_str();
}

/******DemoPluginCreator*******/
DemoPluginCreator::DemoPluginCreator()
{
    log(" run here now! ");
    m_fc.nbFields = 0;
    m_fc.fields = nullptr;
    m_plugin_name = "demo_plugin";
    m_plugin_version = "01";
}

const char* DemoPluginCreator::getPluginName() const
{
    log(" run here now! ");
    return m_plugin_name.c_str();
}

const char* DemoPluginCreator::getPluginVersion() const
{
    log(" run here now! ");
    return m_plugin_version.c_str();
}

const nvinfer1::PluginFieldCollection* DemoPluginCreator::getFieldNames()
{
    log(" run here now! ");
    return &m_fc;
}

nvinfer1::IPluginV2* DemoPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    log(" run here now! ");
    auto* plugin = new DemoPlugin(name);
    plugin->setPluginNamespace(m_namespace.c_str());
    return plugin;
}

nvinfer1::IPluginV2* DemoPluginCreator::deserializePlugin(const char* name, 
    const void* serialData, size_t serialLength)
{
    log(" run here now! ");
    return new DemoPlugin(name, serialData, serialLength);
}

void DemoPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    log(" run here now! ");
    m_namespace = pluginNamespace;
}

const char* DemoPluginCreator::getPluginNamespace() const
{
    log(" run here now! ");
    return m_namespace.c_str();
}

/**
 * 注册plugin ！！！！！！！！！！！！！！
*/
REGISTER_TENSORRT_PLUGIN(DemoPluginCreator);