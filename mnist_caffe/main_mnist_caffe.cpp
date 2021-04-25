/**
 * @file        - main_mnist_caffe.cpp
 * @author      - wdn
 * @brief       - mnist_caffe demp
 * 
 * @copyright Copyright (c) 2021, wdn. All rights reserved.
 * 
 */

#include <iostream>

#include "helper.h"
#include "mnist_caffe.h"

int main()
{
    const std::string caffe_model = "mnist_caffe/model/mnist.caffemodel";
    const std::string caffe_prototxt = "mnist_caffe/model/mnist.prototxt";
    const std::string engine_file = "mnist_caffe/model/mnist.bin";
    int batch_size = 1;
    const std::vector<std::string> output_name{"prob"};
    infer_precision_t precision = INFER_FP32;

    // MnistCaffe(std::string& caffe_prototxt, std::string& caffe_model, std::string& engine_file, int batch_size, 
    //     std::vector<std::string> output_name, infer_precision_t precision) ;
    MnistCaffe* mnist_caffe = new MnistCaffe(caffe_prototxt, caffe_model, engine_file, batch_size, output_name, precision);

    mnist_caffe->build();
    std::string pgm_name = "mnist_caffe/image/9.pgm";
    mnist_caffe->process_input(pgm_name);
    mnist_caffe->infer();
    mnist_caffe->process_result();

    delete mnist_caffe;
    return 0;
}