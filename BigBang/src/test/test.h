#ifndef TEST_H
#define TEST_H

#define BIGBANG_TEST

#include <vector>
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../include/base.h"
#include "../../include/gpu_config.h"
#include "../../include/layer.h"
#include "../../include/log.h"
#include "../../include/tensor.h"
#include "../../include/layers/conv_layer.h"
#include "../../include/layers/inner_product_layer.h"
#include "../../include/layers/pooling_layer.h"
#include "../../include/util/math_function_ptr.h"
using namespace BigBang;

class Test {
public:
	Test() {
		//CUDA_CHECK(cudaSetDevice(0));
	}

	void TestAll() {
		TestTensor_CPU();
		TestTensor_GPU();
		TestConvLayerFeedForward();
		TestConvLayerBackward();
		TestInnerProduct();
		TestInnerProductBackward();
		TestMaxPoolLayerFeedForward();
		TestMaxPoolLayerBackward();
		TestGpuGemm();
	}

private:
	void TestTensor_CPU();
	void TestTensor_GPU();
	void TestConvLayerFeedForward();
	void TestConvLayerBackward();
	void TestInnerProduct();
	void TestInnerProductBackward();
	void TestMaxPoolLayerFeedForward();
	void TestMaxPoolLayerBackward();
	void TestGpuGemm();
};

#endif
