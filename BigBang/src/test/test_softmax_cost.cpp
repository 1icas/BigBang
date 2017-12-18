#include "test.h"


void Test::TestSoftmaxCostLayerBackward_CPU() {
	Tensor<float> bottom(std::vector<int>{3, 1, 3, 1});
	Tensor<float> top;
	float* bottom_data = bottom.mutable_cpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
		bottom_data[i] = i;
	}
	LayerParameter lp;
	SoftmaxCostLayer<float> cost_layer(lp);
	cost_layer.SetUp(&bottom, &top);

	top.mutable_cpu_data()[0] = 1;
	top.mutable_cpu_data()[1] = 0;
	top.mutable_cpu_data()[2] = 2;

	cost_layer.Forward(&bottom, &top);
	cost_layer.Backward(&top, &bottom);
	float true_result[] = { 0.0900306, -0.755272, 0.6652410, -0.909969,  0.244728, 0.665241,  0.0900306,
		0.244728 , -0.3347590 };
	
	for (int i = 0; i < bottom.size(); ++i) {
		CHECK_NEAREST(true_result[i], bottom.cpu_diff_data()[i], 0.00001);
	}
}

void Test::TestSoftmaxCostLayerBackward_GPU() {
	Config::Get().set_mode(Config::ProcessUnit::GPU);
	Tensor<float> bottom(std::vector<int>{3, 1, 3, 1});
	Tensor<float> top;
	float* bottom_data = bottom.mutable_cpu_data();
	for (int i = 0; i < bottom.size(); ++i) {
		bottom_data[i] = i;
	}
	LayerParameter lp;
	SoftmaxCostLayer<float> cost_layer(lp);
	cost_layer.SetUp(&bottom, &top);

	top.mutable_cpu_data()[0] = 1;
	top.mutable_cpu_data()[1] = 0;
	top.mutable_cpu_data()[2] = 2;

	cost_layer.Forward(&bottom, &top);
	cost_layer.Backward(&top, &bottom);
	float true_result[] = { 0.0900306, -0.755272, 0.6652410, -0.909969,  0.244728, 0.665241,  0.0900306,
		0.244728 , -0.3347590 };
	for (int i = 0; i < bottom.size(); ++i) {
		CHECK_NEAREST(true_result[i], bottom.cpu_diff_data()[i], 0.00001);
	}
	Config::Get().set_mode(Config::ProcessUnit::CPU);

}
