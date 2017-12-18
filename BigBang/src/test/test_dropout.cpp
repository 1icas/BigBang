#include "test.h"

void Test::TestDropoutLayerForward_Backward_CPU() {
	std::cout << "This the dropout cpu test output" << std::endl;
	LayerParameter lp;
	auto dropout_params = lp.mutable_dropout_layer_param();
	dropout_params->set_dropout_ratio(0.8);
	Tensor<float> bottom(std::vector<int>{20, 1, 1, 1});
	Tensor<float> top;
	float* mutable_bottom_data = bottom.mutable_cpu_data();
	for (int i = 0; i < 20; ++i) {
		mutable_bottom_data[i] = 1;
	}
	DropoutLayer<float> dp(lp);
	dp.SetUp(&bottom, &top);
	dp.Forward(&bottom, &top);
	const float* top_data = top.cpu_data();
	for (int i = 0; i < 20; ++i) {
		std::cout << top_data[i] << std::endl;
	}
	std::cout << "----------------------" << std::endl;

	float* mutable_top_diff_data = top.mutable_cpu_diff_data();
	for (int i = 0; i < 20; ++i) {
		mutable_top_diff_data[i] = 1;
	}
	dp.Backward(&top, &bottom);
	const float* bottom_diff_data = bottom.cpu_diff_data();
	for (int i = 0; i < 20; ++i) {
		std::cout << bottom_diff_data[i] << std::endl;
	}
}

void Test::TestDropoutLayerForward_Backward_GPU() {
	std::cout << "This the dropout gpu test output" << std::endl;
	Config::Get().set_mode(Config::ProcessUnit::GPU);
	LayerParameter lp;
	auto dropout_params = lp.mutable_dropout_layer_param();
	dropout_params->set_dropout_ratio(0.5);
	Tensor<float> bottom(std::vector<int>{20, 1, 1, 1});
	Tensor<float> top;
	float* mutable_bottom_data = bottom.mutable_cpu_data();
	for (int i = 0; i < 20; ++i) {
		mutable_bottom_data[i] = 1;
	}
	DropoutLayer<float> dp(lp);
	dp.SetUp(&bottom, &top);
	dp.Forward(&bottom, &top);
	const float* top_data = top.cpu_data();
	for (int i = 0; i < 20; ++i) {
		std::cout << top_data[i] << std::endl;
	}
	std::cout << "----------------------" << std::endl;

	float* mutable_top_diff_data = top.mutable_cpu_diff_data();
	for (int i = 0; i < 20; ++i) {
		mutable_top_diff_data[i] = 1;
	}
	dp.Backward(&top, &bottom);
	const float* bottom_diff_data = bottom.cpu_diff_data();
	for (int i = 0; i < 20; ++i) {
		std::cout << bottom_diff_data[i] << std::endl;
	}
	Config::Get().set_mode(Config::ProcessUnit::CPU);

}