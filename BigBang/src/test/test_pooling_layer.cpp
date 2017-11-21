#include "test.h"

void Output(Tensor<float>& output, bool use_gpu) {
	LayerParameter l_p;
	auto pool_param = l_p.mutable_pooling_layer_param();
	pool_param->set_kernel_h(2);
	pool_param->set_kernel_w(2);
	pool_param->set_stride_h(2);
	pool_param->set_stride_w(2);
	PoolingLayer<float> pooling_layer(l_p);

	Tensor<float> input(std::vector<int>{1, 1, 4, 4});
	//Tensor<float> output(std::vector<int>{1, 1, 2, 2});

	float* input_data = input.mutable_cpu_data();
	input_data[0] = 1;
	input_data[1] = 6;
	input_data[2] = 2;
	input_data[3] = 4;
	input_data[4] = 10;
	input_data[5] = 8;
	input_data[6] = 9;
	input_data[7] = 6;
	input_data[8] = 16;
	input_data[9] = 2;
	input_data[10] = 4;
	input_data[11] = 8;
	input_data[12] = 10;
	input_data[13] = 15;
	input_data[14] = 2;
	input_data[15] = 6;
	pooling_layer.SetUp(&input, &output);
	pooling_layer.Forward(&input, &output);
}
//input 2*1*4*4
//pool 2*2
//pool_stride 2*2
void Test::TestMaxPoolLayerFeedForward() {
	Tensor<float> output(std::vector<int>{1, 1, 2, 2});
	Output(output, false);
	const float* output_data = output.cpu_data();
	float true_result[] = { 10, 9, 16, 8 };
	for (int i = 0; i < 4; ++i) {
		CHECK_EQ(output_data[i], true_result[i]);
	}

	Tensor<float> output1(std::vector<int>{1, 1, 2, 2});
	Output(output1, true);
	const float* output1_data = output1.cpu_data();
	for (int i = 0; i < 4; ++i) {
		//std::cout << output1_data[i] << std::endl;
		CHECK_EQ(output1_data[i], true_result[i]);
	}
}

void Test::TestMaxPoolLayerBackward() {
	LayerParameter l_p;
	auto pool_param = l_p.mutable_pooling_layer_param();
	pool_param->set_kernel_h(2);
	pool_param->set_kernel_w(2);
	pool_param->set_stride_h(2);
	pool_param->set_stride_w(2);
	PoolingLayer<float> pooling_layer(l_p);
	Tensor<float> input(std::vector<int>{1, 1, 4, 4});
	Tensor<float> output(std::vector<int>{1, 1, 2, 2});
	float* input_data = input.mutable_cpu_data();
	input_data[0] = 1;
	input_data[1] = 6;
	input_data[2] = 2;
	input_data[3] = 4;
	input_data[4] = 10;
	input_data[5] = 8;
	input_data[6] = 9;
	input_data[7] = 6;
	input_data[8] = 16;
	input_data[9] = 2;
	input_data[10] = 4;
	input_data[11] = 8;
	input_data[12] = 10;
	input_data[13] = 15;
	input_data[14] = 2;
	input_data[15] = 6;
	pooling_layer.SetUp(&input, &output);
	pooling_layer.Forward(&input, &output);
	float* output_diff_data = output.mutable_cpu_diff_data();
	for (int i = 0; i < 4; ++i) {
		output_diff_data[i] = output.cpu_data()[i];
	}
	pooling_layer.Backward(&output, &input);
	const float* diff_data = input.cpu_diff_data();
	for (int i = 0; i < 16; ++i) {
		std::cout << diff_data[i] << std::endl;
	}

	LayerParameter l_p1;
	auto pool_param1 = l_p1.mutable_pooling_layer_param();
	pool_param1->set_kernel_h(2);
	pool_param1->set_kernel_w(2);
	pool_param1->set_stride_h(2);
	pool_param1->set_stride_w(2);
	PoolingLayer<float> pooling_layer1(l_p1);
	Tensor<float> input1(std::vector<int>{1, 1, 4, 4});
	Tensor<float> output1(std::vector<int>{1, 1, 2, 2});
	float* input_data1 = input1.mutable_cpu_data();
	input_data1[0] = 1;
	input_data1[1] = 6;
	input_data1[2] = 2;
	input_data1[3] = 4;
	input_data1[4] = 10;
	input_data1[5] = 8;
	input_data1[6] = 9;
	input_data1[7] = 6;
	input_data1[8] = 16;
	input_data1[9] = 2;
	input_data1[10] = 4;
	input_data1[11] = 8;
	input_data1[12] = 10;
	input_data1[13] = 15;
	input_data1[14] = 2;
	input_data1[15] = 6;
	pooling_layer1.SetUp(&input1, &output1);
	pooling_layer1.Forward(&input1, &output1);
	float* output_diff_data1 = output1.mutable_cpu_diff_data();
	for (int i = 0; i < 4; ++i) {
		output_diff_data1[i] = output1.cpu_data()[i];
	}
	pooling_layer1.Backward(&output1, &input1);
	const float* diff_data1 = input1.cpu_diff_data();
	for (int i = 0; i < 16; ++i) {
		std::cout << diff_data1[i] << std::endl;
	}




}
