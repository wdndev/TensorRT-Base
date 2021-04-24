/**
 * @file        - demo_plugin.cpp
 * @author      - wdn (you@domain.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include "NMS_V3.h"
#include "helper.h"


// log
#define log(...) {  \
    char str[100];  \
    sprintf(str, __VA_ARGS__);  \
    helper::debug << "(@ _ @) Here ---> call " << "[" << __FILE__    \
                  << __FUNCTION__ << "][Line " << __LINE__ << "]"   \
                  << str << std::endl;                              \
}

NMSPlugin::NMSPlugin(const std::string layer_name) : m_layer_name(layer_name)
{
    log(" run here now! ");
    m_plugin_name = "NonMaxSuppressionV3";
    m_plugin_version = "1";
    m_namespace = "MY_TRT";
}

NMSPlugin::NMSPlugin(const std::string layer_name, const void* data, size_t length)
    : m_layer_name(layer_name)
{
    log(" run here now! ");
}

const char* NMSPlugin::getPluginType() const
{
    log(" run here now! ");
    return m_plugin_version.c_str();
}

const char* NMSPlugin::getPluginVersion() const
{
    log(" run here now! ");
    return m_plugin_name.c_str();
}

int NMSPlugin::getNbOutputs() const
{
    log(" run here now! ");
    return 1;
}

nvinfer1::Dims NMSPlugin::getOutputDimensions(int index, 
    const nvinfer1::Dims* inputs, int nbInputDims)
{
    log(" run here now! ");
    return nvinfer1::Dims3(inputs[1].d[1],inputs[1].d[2],inputs[1].d[3]);
}

bool NMSPlugin::supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const
{
    log(" run here now! ");
    return false;
}

int NMSPlugin::initialize()
{
    log(" run here now! ");
    return 0;
}

void NMSPlugin::terminate()
{
    log(" run here now! ");
}

size_t NMSPlugin::getWorkspaceSize(int maxBatchSize) const
{
    log(" run here now! ");
    return 0;
}

int NMSPlugin::enqueue(int batchSize, const void* const* inputs, 
            void** outputs, void* workspace, cudaStream_t stream)
{
    log(" run here now! ");
    return 0;
}

size_t NMSPlugin::getSerializationSize() const
{
    log(" run here now! ");
    return 0;
}

void NMSPlugin::serialize(void* buffer) const
{
    log(" run here now! ");
}

nvinfer1::DataType NMSPlugin::getOutputDataType(int index, 
    const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    log(" run here now! ");
    return nvinfer1::DataType::kFLOAT;
}

bool NMSPlugin::isOutputBroadcastAcrossBatch(int outputIndex, 
    const bool* inputIsBroadcasted, int nbInputs) const
{
    log(" run here now! ");
    return false;
}

bool NMSPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    log(" run here now! ");
    return false;
}

void NMSPlugin::configurePlugin(const nvinfer1::Dims* inputDims, 
    int nbInputs, const nvinfer1::Dims* outputDims, int nbOutputs, 
    const nvinfer1::DataType* inputTypes, const nvinfer1::DataType* outputTypes,
    const bool* inputIsBroadcast, const bool* outputIsBroadcast, 
    nvinfer1::PluginFormat floatFormat, int maxBatchSize)
{
    log(" run here now! ");
}

void NMSPlugin::destroy()
{
    log(" run here now! ");
}

nvinfer1::IPluginV2Ext* NMSPlugin::clone() const
{
    log(" run here now! ");
    auto* plugin = new NMSPlugin(m_layer_name);
    plugin->setPluginNamespace(m_namespace.c_str());
    return plugin;
}

void NMSPlugin::setPluginNamespace(const char* pluginNamespace)
{
    log(" run here now! ");
}

const char* NMSPlugin::getPluginNamespace() const
{
    log(" run here now! ");
    return m_namespace.c_str();
}

/******NMSPluginCreator*******/
NMSPluginCreator::NMSPluginCreator()
{
    log(" run here now! ");
    m_fc.nbFields = 0;
    m_fc.fields = nullptr;
    m_plugin_name = "NonMaxSuppressionV3";
    m_plugin_version = "1";
    m_namespace = "MY_TRT";
}

const char* NMSPluginCreator::getPluginName() const
{
    log(" run here now! ");
    return m_plugin_name.c_str();
}

const char* NMSPluginCreator::getPluginVersion() const
{
    log(" run here now! ");
    return m_plugin_version.c_str();
}

const nvinfer1::PluginFieldCollection* NMSPluginCreator::getFieldNames()
{
    log(" run here now! ");
    return &m_fc;
}

nvinfer1::IPluginV2* NMSPluginCreator::createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc)
{
    log(" run here now! ");
    auto* plugin = new NMSPlugin(name);
    plugin->setPluginNamespace(m_namespace.c_str());
    return plugin;
}

nvinfer1::IPluginV2* NMSPluginCreator::deserializePlugin(const char* name, 
    const void* serialData, size_t serialLength)
{
    log(" run here now! ");
    return new NMSPlugin(name, serialData, serialLength);
}

void NMSPluginCreator::setPluginNamespace(const char* pluginNamespace)
{
    log(" run here now! ");
    m_namespace = pluginNamespace;
}

const char* NMSPluginCreator::getPluginNamespace() const
{
    log(" run here now! ");
    return m_namespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(NMSPluginCreator);