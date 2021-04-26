# Plugin

## 1.Description

TensorRT已经只支持了许多常见的神经网络层,比如卷积, 池化, BN等等. 但是依然还有很多操作和算子是不支持的,所以TensorRT提供了接口让我们可以编写插件来实现自己的自定义层。

随着tensorRT的不断发展(v5->v6->v7)，TensorRT的插件的使用方式也在不断更新。插件接口也在不断地变化，由v5版本的`IPluginV2Ext`，到v6版本的`IPluginV2IOExt`和`IPluginV2DynamicExt`。未来不知道会不会出来新的API，不过这也不是咱要考虑的问题，因为TensorRT的后兼容性做的很好，根本不用担心你写的旧版本插件在新版本上无法运行。

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