#ifndef TEST_H
#define TEST_H

#define BIGBANG_TEST

#include <vector>
#include <memory>
#include <iostream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../include/base.h"
#include "../../include/config.h"
//#include "../../include/gpu_config.h"
#include "../../include/layer.h"
#include "../../include/log.h"
#include "../../include/tensor.h"
#include "../../include/layers/conv_layer.h"
#include "../../include/layers/data_layer.h"
#include "../../include/layers/dropout_layer.h"
#include "../../include/layers/inner_product_layer.h"
#include "../../include/layers/mse_layer.h"
#include "../../include/layers/pooling_layer.h"
#include "../../include/layers/sigmoid_layer.h"
#include "../../include/layers/softmax_cost_layer.h"
#include "../../include/util/data_reader.h"
#include "../../include/util/common.h"
#include "../../include/util/image_common.h"
#include "../../include/util/math_function_ptr.h"
#include "../../include/util/parse.h"
#include "../../include/solver.h"

#include "../../proto/bigbang.pb.h"


using namespace BigBang;

class Test {
public:
	Test() {
		//CUDA_CHECK(cudaSetDevice(0));
	}

	void TestAll() {
		/*TestTensor_CPU();
		TestTensor_GPU();
		TestConvLayerFeedForward_CPU();
		TestConvLayerBackward_CPU();
		TestInnerProduct();
		TestInnerProductBackward();
		TestMaxPoolLayerFeedForward();
		TestMaxPoolLayerBackward();
		TestGpuGemm();
		TestSoftmaxCostLayerBackward_CPU();
		TestSoftmaxCostLayerBackward_GPU();*/
		//TestDropoutLayerForward_Backward_CPU();
		//TestDropoutLayerForward_Backward_GPU();
		//TestFullyConnectLayer<double>();
		//TestConvLayerFeedForward_GPU();
		//TestConvLayerBackward_GPU();
		//TestConvLayerNetwork<double>();
		//TestConvLayerNetwork1<double>();
		//TestLoadDataAsync();
		//TestParseTextFileToProtobuf();
		TestCifar10Model<double>();
	}


private:
	void TestTensor_CPU();
	void TestTensor_GPU();
	void TestConvLayerFeedForward_CPU();
	void TestConvLayerFeedForward_GPU();
	void TestConvLayerBackward_CPU();
	void TestConvLayerBackward_GPU();
	void TestSoftmaxCostLayerBackward_CPU();
	void TestSoftmaxCostLayerBackward_GPU();
	void TestDropoutLayerForward_Backward_CPU();
	void TestDropoutLayerForward_Backward_GPU();


	void TestInnerProduct();
	void TestInnerProductBackward();
	void TestMaxPoolLayerFeedForward();
	void TestMaxPoolLayerBackward();
	void TestGpuGemm();
	void TestLoadDataAsync();
	void TestParseTextFileToProtobuf();
	template<typename dtype>
	void TestFullyConnectLayer();
	template<typename dtype>
	void TestConvLayerNetwork();
	template<typename dtype>
	void TestCifar10Model();
	
};

#endif
