/**
 * @file        - plugin_factory.cpp
 * @author      - wdn (you@domain.com)
 * @brief 
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#include "plugin_factory.h"
#include "demo_plugin/demo_plugin.h"
#include "NMS_V3/NMS_V3.h"

PluginFactory::PluginFactory(plugin_param_t param)
{
    helper::debug << "[Plugin] : create plugin factory";
    m_plugin_param.test = param.test;
    m_support_plugin = {"demo_plugin", "prelu", "NonMaxSuppressionV3"};
}

PluginFactory::PluginFactory()
{
    helper::debug << "[Plugin] : create plugin factory";
    m_plugin_param.test = 1111;
    m_support_plugin = {"demo_plugin", "prelu"};
}

bool PluginFactory::is_plugin(const std::string layer_name)
{
    for(auto& s : m_support_plugin)
    {
        if (s == layer_name)
            return true;
    }

    return false;
}

nvinfer1::IPluginV2* PluginFactory::create_plugin(const std::string& layer_name, 
        const nvinfer1::Weights* weights, int nb_weights, const std::string& lib_namespace)
{
    if(!is_plugin(layer_name))
    {
        helper::err << "[Plugin] : Unsupport " << layer_name << " plugin, please coding this plugin" << std::endl;
        return nullptr;
    }

    if(layer_name == "demo_plugin")
    {
        return (nvinfer1::IPluginV2*)(new nvinfer1::plugin::DemoPlugin(layer_name));
        //nvinfer1::plugin::DemoPlugin
    }
    else if(layer_name == "NonMaxSuppressionV3")
    {
        //return (nvinfer1::IPluginV2*)(new NMSPlugin(layer_name));
        return (nvinfer1::IPluginV2*)(new nvinfer1::plugin::DemoPlugin(layer_name));
    }
    else if(layer_name == "")
    {
        return nullptr;
    }
    
}

nvinfer1::IPluginV2* PluginFactory::create_plugin(std::string layer_name, 
        const void* serial_data, size_t serial_length)
{
    if(!is_plugin(layer_name))
    {
        helper::err << "[Plugin] : Unsupport " << layer_name << " plugin, please coding this plugin" << std::endl;
        return nullptr;
    }

    if(layer_name == "demo_plugin")
    {
        return (nvinfer1::IPluginV2*)(new nvinfer1::plugin::DemoPlugin(layer_name, serial_data, serial_length));
    }
    // else if(layer_name == "prelu")
    // {
    //     return (nvinfer1::IPlugin*)(new PReLUPlugin(serial_data, serial_length));
    // }
    else if(layer_name == "")
    {
        return nullptr;
    }
}