#include "test.h"

void Output(Tensor<float>& output) {
	PoolingLayerParams<float> plp(PoolingLayerParams<float>::Pool::MaxPool, 2, 2, 2, 2);
	LayerParamsManage<float> manage;
	manage.type_ = "Pooling";
	manage.use_gpu_ = false;
	manage.pooling_layer_params_ = plp;
	PoolingLayer<float> pooling_layer(manage);

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
	Output(output);
	const float* output_data = output.cpu_data();
	float true_result[] = { 10, 9, 16, 8 };
	for (int i = 0; i < 4; ++i) {
		CHECK_EQ(output_data[i], true_result[i]);
	}
}

void Test::TestMaxPoolLayerBackward() {
	PoolingLayerParams<float> plp(PoolingLayerParams<float>::Pool::MaxPool, 2, 2, 2, 2);
	LayerParamsManage<float> manage;
	manage.type_ = "Pooling";
	manage.use_gpu_ = false;
	manage.pooling_layer_params_ = plp;
	PoolingLayer<float> pooling_layer(manage);
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
}
