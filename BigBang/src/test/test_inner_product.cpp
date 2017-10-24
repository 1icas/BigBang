#include "test.h"

void Test::TestInnerProduct() {
	std::shared_ptr<Tensor<float>> weights(new Tensor<float>(std::vector<int>{1, 1, 3, 2}));
	std::shared_ptr<Tensor<float>> biases(new Tensor<float>(std::vector<int>{1, 1, 2, 1}));
	Tensor<float> input(std::vector<int>{4, 1, 1, 3});
	float* input_data = input.mutable_cpu_data();
	for (int i = 1; i < 13; ++i) {
		input_data[i - 1] = i;
	}
	float* weights_data = weights->mutable_cpu_data();
	float* biases_data = biases->mutable_cpu_data();
	weights_data[0] = 1;
	weights_data[1] = 2;
	weights_data[2] = 2;
	weights_data[3] = 1;
	weights_data[4] = 2;
	weights_data[5] = 2;
	biases_data[0] = 1;
	biases_data[1] = 2;

	InnerProductLayerParams<float> params(1.0, 1.0, true, FillerParams<float>(),
		FillerParams<float>());
	params.weights_ = weights;
	params.biases_ = biases;
	LayerParamsManage<float> manage;
	manage.use_gpu_ = false;
	manage.type_ = "InnerProduct";
	manage.inner_product_layer_params_ = params;
	InnerProductLayer<float> layer(manage);
	Tensor<float> output(std::vector<int>{4, 1, 1, 2});
	layer.SetUp(&input, &output);
	layer.Forward(&input, &output);
	const float* output_data = output.cpu_data();
	const float true_result[] = { 12, 12, 27, 27, 42, 42, 57, 57 };
	for (int i = 0; i < 8; ++i)
		CHECK_EQ(true_result[i], output_data[i]);
}

void Test::TestInnerProductBackward() {
	std::shared_ptr<Tensor<float>> weights(new Tensor<float>(std::vector<int>{1, 1, 2, 3}));
	std::shared_ptr<Tensor<float>> biases(new Tensor<float>(std::vector<int>{1, 1, 3, 1}));

	float* weights_data = weights->mutable_cpu_data();
	float* biases_data = biases->mutable_cpu_data();
	Tensor<float> input(std::vector<int>{1, 1, 1, 2});
	float* input_data = input.mutable_cpu_data();
	input_data[0] = 1;
	input_data[1] = 2;
	weights_data[0] = 2;
	weights_data[1] = 3;
	weights_data[2] = 2;
	weights_data[3] = 1;
	weights_data[4] = 2;
	weights_data[5] = 5;
	biases_data[0] = 1;
	biases_data[1] = 2;
	biases_data[2] = 3;

	InnerProductLayerParams<float> params(1.0, 1.0, true, FillerParams<float>(),
		FillerParams<float>());
	params.weights_ = weights;
	params.biases_ = biases;
	LayerParamsManage<float> manage;
	manage.use_gpu_ = false;
	manage.type_ = "InnerProduct";
	manage.inner_product_layer_params_ = params;
	InnerProductLayer<float> layer(manage);
	Tensor<float> output(std::vector<int>{1, 1, 1, 3});
	layer.SetUp(&input, &output);
	layer.Forward(&input, &output);
	float* output_data = output.mutable_cpu_data();
	float true_result[] = { 5, 9, 15 };
	for (int i = 0; i < 3; ++i) {
		CHECK_EQ(output_data[i], true_result[i]);
	}
	float* output_diff_data = output.mutable_cpu_diff_data();
	for (int i = 0; i < 3; ++i) {
		output_diff_data[i] = 0.1*(i + 1);
	}
	layer.Backward(&output, &input);
	const float* input_diff_data = input.cpu_diff_data();
	float true_result1[] = { 1.4, 2.0 };
	for (int i = 0; i < 2; ++i) {
		if (input_diff_data[i] - true_result1[i] > 0.001) {
			std::cout << __LINE__ << std::endl;
			std::cout << __FILE__ << std::endl;
		}
	}
	std::vector<std::shared_ptr<Tensor<float>>> hyper_params = layer.GetParams();
	const float* weights_diff_data = hyper_params[0]->cpu_diff_data();
	for (int i = 0; i < 6; ++i) {
		std::cout << weights_diff_data[i] << std::endl;
	}
}