#include "test.h"

template<typename dtype>
class TestConvLayer : public ConvLayer<dtype> {
public:
	TestConvLayer(const LayerParameter& params)
		: ConvLayer(params){}

	/*virtual void SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) override {
		ConvLayer<dtype>::SetUp(bottom, top);
		dtype* kernel_data = kernels_->mutable_cpu_data();
		kernel_data[0] = 1;
		kernel_data[1] = 0;
		kernel_data[2] = 1;
		kernel_data[3] = 2;
		kernel_data[4] = 3;
		kernel_data[5] = 2;
		kernel_data[6] = 1;
		kernel_data[7] = 1;
		kernel_data[8] = 1;
		kernel_data[9] = 2;
		kernel_data[10] = 3;
		kernel_data[11] = 2;
		kernel_data[12] = 6;
		kernel_data[13] = 1;
		kernel_data[14] = 0;
		kernel_data[15] = 0;
		kernel_data[16] = 0;
		kernel_data[17] = 2;
		kernel_data[18] = 2;
		kernel_data[19] = 2;
		kernel_data[20] = 0;
		kernel_data[21] = 5;
		kernel_data[22] = 1;
		kernel_data[23] = 0;
		kernel_data[24] = 3;
		kernel_data[25] = 4;
		kernel_data[26] = 1;
		dtype* bias_data = biases_->mutable_cpu_data();
		biases_data[0] = 1;
		biases_data[1] = 2;
		biases_data[2] = 3;
	}*/

	void set_kernel(const std::shared_ptr<Tensor<dtype>>& kernel) {
		kernels_ = kernel;
	}

	void set_bias(const std::shared_ptr<Tensor<dtype>>& bias) {
		biases_ = bias;
	}

};


void Test::TestConvLayerFeedForward_CPU() {
	Tensor<float> input(std::vector<int>{2, 1, 5, 5});
	Tensor<float> output(std::vector<int>{2, 3, 2, 2});

	float* input_data = input.mutable_cpu_data();
	for (int i = 0; i < 25; ++i) {
		input_data[i] = i + 1;
	}
	for (int i = 0; i < 25; ++i) {
		input_data[i + 25] = 25 - i;
	}

	std::shared_ptr<Tensor<float>> kernel = std::make_shared<Tensor<float>>(std::vector<int>{3, 1, 3, 3});
	std::shared_ptr<Tensor<float>> biases = std::make_shared<Tensor<float>>(std::vector<int>{3, 1, 1, 1});

	float* kernel_data = kernel->mutable_cpu_data();
	kernel_data[0] = 1;
	kernel_data[1] = 0;
	kernel_data[2] = 1;
	kernel_data[3] = 2;
	kernel_data[4] = 3;
	kernel_data[5] = 2;
	kernel_data[6] = 1;
	kernel_data[7] = 1;
	kernel_data[8] = 1;
	kernel_data[9] = 2;
	kernel_data[10] = 3;
	kernel_data[11] = 2;
	kernel_data[12] = 6;
	kernel_data[13] = 1;
	kernel_data[14] = 0;
	kernel_data[15] = 0;
	kernel_data[16] = 0;
	kernel_data[17] = 2;
	kernel_data[18] = 2;
	kernel_data[19] = 2;
	kernel_data[20] = 0;
	kernel_data[21] = 5;
	kernel_data[22] = 1;
	kernel_data[23] = 0;
	kernel_data[24] = 3;
	kernel_data[25] = 4;
	kernel_data[26] = 1;
	float* biases_data = biases->mutable_cpu_data();
	biases_data[0] = 1;
	biases_data[1] = 2;
	biases_data[2] = 3;

	LayerParameter l_p;
	auto conv_params = l_p.mutable_conv_layer_param();
	conv_params->set_kernel_groups(3);
	conv_params->set_kernel_channels(1);
	conv_params->set_kernel_h(3);
	conv_params->set_kernel_w(3);
	conv_params->set_pad_h(0);
	conv_params->set_pad_w(0);
	conv_params->set_stride_h(2);
	conv_params->set_stride_w(2);
	conv_params->set_use_bias(true);
	auto k_f = conv_params->mutable_kernel_filler();
	k_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);
	auto b_f = conv_params->mutable_bias_filler();
	b_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);

	TestConvLayer<float> conv_layer(l_p);
	conv_layer.SetUp(&input, &output);
	conv_layer.set_kernel(kernel);
	conv_layer.set_bias(biases);
	conv_layer.Forward(&input, &output);

	const float* output_data = output.cpu_data();
	float true_result[] = {
		90, 114, 210, 234,
		85, 117, 245, 277,
		140, 176, 320, 356,
		224, 200, 104, 80,
		335, 303, 175, 143,
		334, 298,	154, 118
	};

	for (int i = 0; i < 24; ++i) {
		CHECK_EQ(true_result[i], output_data[i]);
		//if (true_result[i] != output_data[i]) std::cout << "wrong result" << std::endl;
	}
}

void Test::TestConvLayerFeedForward_GPU() {
	Config::Get().set_mode(Config::ProcessUnit::GPU);

	Tensor<float> input(std::vector<int>{2, 1, 5, 5});
	Tensor<float> output(std::vector<int>{2, 3, 2, 2});

	float* input_data = input.mutable_cpu_data();
	for (int i = 0; i < 25; ++i) {
		input_data[i] = i + 1;
	}
	for (int i = 0; i < 25; ++i) {
		input_data[i + 25] = 25 - i;
	}

	std::shared_ptr<Tensor<float>> kernel = std::make_shared<Tensor<float>>(std::vector<int>{3, 1, 3, 3});
	std::shared_ptr<Tensor<float>> biases = std::make_shared<Tensor<float>>(std::vector<int>{3, 1, 1, 1});

	float* kernel_data = kernel->mutable_cpu_data();
	kernel_data[0] = 1;
	kernel_data[1] = 0;
	kernel_data[2] = 1;
	kernel_data[3] = 2;
	kernel_data[4] = 3;
	kernel_data[5] = 2;
	kernel_data[6] = 1;
	kernel_data[7] = 1;
	kernel_data[8] = 1;
	kernel_data[9] = 2;
	kernel_data[10] = 3;
	kernel_data[11] = 2;
	kernel_data[12] = 6;
	kernel_data[13] = 1;
	kernel_data[14] = 0;
	kernel_data[15] = 0;
	kernel_data[16] = 0;
	kernel_data[17] = 2;
	kernel_data[18] = 2;
	kernel_data[19] = 2;
	kernel_data[20] = 0;
	kernel_data[21] = 5;
	kernel_data[22] = 1;
	kernel_data[23] = 0;
	kernel_data[24] = 3;
	kernel_data[25] = 4;
	kernel_data[26] = 1;
	float* biases_data = biases->mutable_cpu_data();
	biases_data[0] = 1;
	biases_data[1] = 2;
	biases_data[2] = 3;


	LayerParameter l_p;
	auto conv_params = l_p.mutable_conv_layer_param();
	conv_params->set_kernel_groups(3);
	conv_params->set_kernel_channels(1);
	conv_params->set_kernel_h(3);
	conv_params->set_kernel_w(3);
	conv_params->set_pad_h(0);
	conv_params->set_pad_w(0);
	conv_params->set_stride_h(2);
	conv_params->set_stride_w(2);
	conv_params->set_use_bias(true);
	auto k_f = conv_params->mutable_kernel_filler();
	k_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);
	auto b_f = conv_params->mutable_bias_filler();
	b_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);

	TestConvLayer<float> conv_layer(l_p);
	conv_layer.SetUp(&input, &output);
	conv_layer.set_kernel(kernel);
	conv_layer.set_bias(biases);
	conv_layer.Forward(&input, &output);

	//conv_layer.FeedForward(&input, &output);

	const float* output_data = output.cpu_data();
	float true_result[] = {
		90, 114, 210, 234,
		85, 117, 245, 277,
		140, 176, 320, 356,
		224, 200, 104, 80,
		335, 303, 175, 143,
		334, 298,	154, 118
	};

	for (int i = 0; i < 24; ++i) {
		std::cout << output_data[i] << std::endl;
		//if (true_result[i] != output_data[i]) std::cout << "wrong result" << std::endl;
	}
	Config::Get().set_mode(Config::ProcessUnit::CPU);
}

void Test::TestConvLayerBackward_CPU() {
	Tensor<float> input(std::vector<int>{1, 3, 4, 4});
	Tensor<float> output(std::vector<int>{1, 1, 3, 3});
	float* input_data = input.mutable_cpu_data();
	input_data[0] = 1;
	input_data[1] = 2;
	input_data[2] = 3;
	input_data[3] = 2;
	input_data[4] = 4;
	input_data[5] = 5;
	input_data[6] = 2;
	input_data[7] = 1;
	input_data[8] = 6;
	input_data[9] = 2;
	input_data[10] = 2;
	input_data[11] = 1;
	input_data[12] = 2;
	input_data[13] = 3;
	input_data[14] = 2;
	input_data[15] = 2;
	input_data[16] = 2;
	input_data[17] = 1;
	input_data[18] = 6;
	input_data[19] = 2;
	input_data[20] = 3;
	input_data[21] = 2;
	input_data[22] = 1;
	input_data[23] = 5;
	input_data[24] = 4;
	input_data[25] = 8;
	input_data[26] = 2;
	input_data[27] = 0;
	input_data[28] = 2;
	input_data[29] = 5;
	input_data[30] = 5;
	input_data[31] = 1;
	input_data[32] = 2;
	input_data[33] = 2;
	input_data[34] = 3;
	input_data[35] = 2;
	input_data[36] = 1;
	input_data[37] = 5;
	input_data[38] = 2;
	input_data[39] = 0;
	input_data[40] = 4;
	input_data[41] = 3;
	input_data[42] = 5;
	input_data[43] = 2;
	input_data[44] = 2;
	input_data[45] = 1;
	input_data[46] = 6;
	input_data[47] = 2;
	std::shared_ptr<Tensor<float>> kernel = std::make_shared<Tensor<float>>(std::vector<int>{1, 3, 2, 2});
	std::shared_ptr<Tensor<float>> biases = std::make_shared<Tensor<float>>(std::vector<int>{1, 3, 1, 1});
	float* kernel_data = kernel->mutable_cpu_data();
	kernel_data[0] = 1;
	kernel_data[1] = 2;
	kernel_data[2] = 2;
	kernel_data[3] = 3;
	kernel_data[4] = 2;
	kernel_data[5] = 3;
	kernel_data[6] = 2;
	kernel_data[7] = 2;
	kernel_data[8] = 1;
	kernel_data[9] = 5;
	kernel_data[10] = 2;
	kernel_data[11] = 6;
	float* biases_data = biases->mutable_cpu_data();
	biases_data[0] = 1;
	biases_data[1] = 2;
	biases_data[2] = 2;

	LayerParameter l_p;
	auto conv_params = l_p.mutable_conv_layer_param();
	conv_params->set_kernel_groups(1);
	conv_params->set_kernel_channels(3);
	conv_params->set_kernel_h(2);
	conv_params->set_kernel_w(2);
	conv_params->set_pad_h(0);
	conv_params->set_pad_w(0);
	conv_params->set_stride_h(1);
	conv_params->set_stride_w(1);
	conv_params->set_use_bias(true);
	auto k_f = conv_params->mutable_kernel_filler();
	k_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);
	auto b_f = conv_params->mutable_bias_filler();
	b_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);

	TestConvLayer<float> conv_layer(l_p);
	conv_layer.SetUp(&input, &output);
	conv_layer.set_kernel(kernel);
	conv_layer.set_bias(biases);
	conv_layer.Forward(&input, &output);

	const float* output_data = output.cpu_data();
	float true_result[] = {
		94, 94, 66,
		125, 102, 61,
		103, 131, 74
	};
	for (int i = 0; i < 9; ++i) {
		CHECK_EQ(true_result[i], output_data[i]);
	}
	float* output_diff = output.mutable_cpu_diff_data();
	output_diff[0] = 0.1;
	output_diff[1] = 0.0;
	output_diff[2] = 0.2;
	output_diff[3] = 0.1;
	output_diff[4] = 0.3;
	output_diff[5] = 0.5;
	output_diff[6] = 0.0;
	output_diff[7] = 0.1;
	output_diff[8] = 0.1;

	conv_layer.Backward(&output, &input);

	//for (int i = 0; i < 48; ++i) {
	//	std::cout << input.cpu_diff_data()[i] << std::endl;
	//}
}

void Test::TestConvLayerBackward_GPU() {
	Config::Get().set_mode(Config::ProcessUnit::GPU);

	Tensor<float> input(std::vector<int>{1, 3, 4, 4});
	Tensor<float> output(std::vector<int>{1, 1, 3, 3});
	float* input_data = input.mutable_cpu_data();
	input_data[0] = 1;
	input_data[1] = 2;
	input_data[2] = 3;
	input_data[3] = 2;
	input_data[4] = 4;
	input_data[5] = 5;
	input_data[6] = 2;
	input_data[7] = 1;
	input_data[8] = 6;
	input_data[9] = 2;
	input_data[10] = 2;
	input_data[11] = 1;
	input_data[12] = 2;
	input_data[13] = 3;
	input_data[14] = 2;
	input_data[15] = 2;
	input_data[16] = 2;
	input_data[17] = 1;
	input_data[18] = 6;
	input_data[19] = 2;
	input_data[20] = 3;
	input_data[21] = 2;
	input_data[22] = 1;
	input_data[23] = 5;
	input_data[24] = 4;
	input_data[25] = 8;
	input_data[26] = 2;
	input_data[27] = 0;
	input_data[28] = 2;
	input_data[29] = 5;
	input_data[30] = 5;
	input_data[31] = 1;
	input_data[32] = 2;
	input_data[33] = 2;
	input_data[34] = 3;
	input_data[35] = 2;
	input_data[36] = 1;
	input_data[37] = 5;
	input_data[38] = 2;
	input_data[39] = 0;
	input_data[40] = 4;
	input_data[41] = 3;
	input_data[42] = 5;
	input_data[43] = 2;
	input_data[44] = 2;
	input_data[45] = 1;
	input_data[46] = 6;
	input_data[47] = 2;
	std::shared_ptr<Tensor<float>> kernel = std::make_shared<Tensor<float>>(std::vector<int>{1, 3, 2, 2});
	std::shared_ptr<Tensor<float>> biases = std::make_shared<Tensor<float>>(std::vector<int>{1, 3, 1, 1});
	float* kernel_data = kernel->mutable_cpu_data();
	kernel_data[0] = 1;
	kernel_data[1] = 2;
	kernel_data[2] = 2;
	kernel_data[3] = 3;
	kernel_data[4] = 2;
	kernel_data[5] = 3;
	kernel_data[6] = 2;
	kernel_data[7] = 2;
	kernel_data[8] = 1;
	kernel_data[9] = 5;
	kernel_data[10] = 2;
	kernel_data[11] = 6;
	float* biases_data = biases->mutable_cpu_data();
	biases_data[0] = 1;
	biases_data[1] = 2;
	biases_data[2] = 2;

	LayerParameter l_p;
	auto conv_params = l_p.mutable_conv_layer_param();
	conv_params->set_kernel_groups(1);
	conv_params->set_kernel_channels(3);
	conv_params->set_kernel_h(2);
	conv_params->set_kernel_w(2);
	conv_params->set_pad_h(0);
	conv_params->set_pad_w(0);
	conv_params->set_stride_h(1);
	conv_params->set_stride_w(1);
	conv_params->set_use_bias(true);
	auto k_f = conv_params->mutable_kernel_filler();
	k_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);
	auto b_f = conv_params->mutable_bias_filler();
	b_f->set_type(FillerParameter::GAUSSIAN_DISTRIBUTION);

	TestConvLayer<float> conv_layer(l_p);
	conv_layer.SetUp(&input, &output);
	conv_layer.set_kernel(kernel);
	conv_layer.set_bias(biases);
	conv_layer.Forward(&input, &output);

	const float* output_data = output.cpu_data();
	float true_result[] = {
		94, 94, 66,
		125, 102, 61,
		103, 131, 74
	};
	/*for (int i = 0; i < 9; ++i) {
		CHECK_EQ(true_result[i], output_data[i]);
	}*/
	float* output_diff = output.mutable_cpu_diff_data();
	output_diff[0] = 0.1;
	output_diff[1] = 0.0;
	output_diff[2] = 0.2;
	output_diff[3] = 0.1;
	output_diff[4] = 0.3;
	output_diff[5] = 0.5;
	output_diff[6] = 0.0;
	output_diff[7] = 0.1;
	output_diff[8] = 0.1;

	conv_layer.Backward(&output, &input);

	for (int i = 0; i < 48; ++i) {
		std::cout << input.cpu_diff_data()[i] << std::endl;
	}
	Config::Get().set_mode(Config::ProcessUnit::CPU);
}