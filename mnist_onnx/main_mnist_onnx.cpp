#include <iostream>

#include "helper.h"
#include "mnist_onnx.h"

int main()
{
    const std::string onnx_model = "mnist_onnx/model/modified.onnx";
    const std::string engine_file = "";
    //const std::string engine_file = "mnist_onnx/model/mnist_onnx.bin";
    int batch_size = 1;
    // const std::vector<std::string> output_name{"Plus214_Output_0"};
    const std::vector<std::string> output_name{"identity_out"};
    infer_precision_t precision = INFER_FP32;

    MnistOnnx* mnist_onnx = new MnistOnnx(onnx_model, engine_file, batch_size, output_name, precision);

    mnist_onnx->build();
    std::string pgm_name = "mnist_onnx/image/9.pgm";
    mnist_onnx->process_input(pgm_name);
    mnist_onnx->infer();
    mnist_onnx->process_result();

    delete mnist_onnx;
    return 0;
}