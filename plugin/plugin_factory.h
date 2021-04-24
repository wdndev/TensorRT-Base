/**
 * @file        - plugin_factory.h
 * @author      - wdn (you@domain.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */

#ifndef TRT_PLUGIN_FACTORY_H
#define TRT_PLUGIN_FACTORY_H

#include <map>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvCaffeParser.h>

#include "helper.h"
#include "demo_plugin/demo_plugin.h"
#include "NMS_V3/NMS_V3.h"

typedef struct
{
    int test;

}plugin_param_t;

class PluginFactory //: public nvinfer1::IPluginFactory
{
public:
    PluginFactory(plugin_param_t params);

    PluginFactory();

    virtual ~PluginFactory(){}

    bool is_plugin(const std::string layer_name);

    nvinfer1::IPluginV2* create_plugin(const std::string& layer_name, const nvinfer1::Weights* weights,
                                     int nb_weights, const std::string& lib_namespace);
    nvinfer1::IPluginV2* create_plugin(std::string layer_name, const void* serial_data, size_t serial_length);
    //nvinfer1::IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override;

private:
    plugin_param_t m_plugin_param;
    
    std::vector<std::string> m_support_plugin;
};


#endif // TRT_PLUGIN_FACTORY_H
