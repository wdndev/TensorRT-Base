/**
 * @file        - plugin_faster_rcnn.h
 * @author      - wdn(dongnian.wang@outlook.com)
 * @brief       - plugin_faster_rcnn
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */
#ifndef PLUGIN_FATSER_RCNN_H
#define PLUGIN_FATSER_RCNN_H

#include <assert.h>
#include <cstring>
#include <memory>


#include <NvCaffeParser.h>
#include <NvInferPlugin.h>

const int pooling_wide = 7;
const int pooling_height = 7;
const int feature_stride = 16;
const int pre_nums_top = 6000;
const int nms_max_out = 300;
const int anchors_ration_count = 3;
const int anchors_scale_count = 3;
const float iou_threshold = 0.7f;
const float min_box_size = 16;
const float spatial_scale = 0.0625f;
const float anchors_ratios[anchors_ration_count] = {0.5f, 1.0f, 2.0f};
const float anchors_scales[anchors_scale_count] = {8.0f, 16.0f, 32.0f};

class FRCNNPluginFactory : public nvcaffeparser1::IPluginFactoryV2
{
public:

    virtual nvinfer1::IPluginV2* createPlugin(const char* layerName, const nvinfer1::Weights* weights,
                 int nbWeights, const char* libNamespace) override
    {
        assert(isPluginV2(layerName));
        if(!strcmp(layerName, "RPROIFused"))
        {
            assert(m_plugin_RPROI == nullptr);
            assert(nbWeights == 0 && weights == nullptr);
            m_plugin_RPROI ==  std::unique_ptr<nvinfer1::IPluginV2, decltype(plugin_deleter)>(
                            createRPNROIPlugin(feature_stride, pre_nums_top, nms_max_out,iou_threshold, 
                                min_box_size, spatial_scale, nvinfer1::DimsHW(pooling_height, pooling_wide), 
                                nvinfer1::Weights{nvinfer1::DataType::kFLOAT, anchors_ratios, anchors_ration_count},
                                nvinfer1::Weights{nvinfer1::DataType::kFLOAT, anchors_scales, anchors_scale_count}),
                            plugin_deleter);
            
            m_plugin_RPROI.get()->setPluginNamespace(libNamespace);
            return m_plugin_RPROI.get();
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    bool isPluginV2(const char* name) override
    {
        return !strcmp(name, "RPROIFused");
    }

    void destroyPlugin()
    {
        m_plugin_RPROI.reset();
    }

    void (*plugin_deleter)(nvinfer1::IPluginV2*)
    {
        [](nvinfer1::IPluginV2* ptr) 
        { 
            ptr->destroy(); 
        }
    };

    std::unique_ptr<nvinfer1::IPluginV2, decltype(plugin_deleter)> m_plugin_RPROI{nullptr, plugin_deleter};
};


#endif // PLUGIN_FATSER_RCNN_H