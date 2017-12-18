#include "../include/net.h"

#include "../include/base.h"
#include "../include/layer_factory.h"
#include "../include/layers/layer_type_macro.h"
#include "../include/util/math_function_ptr.h"

namespace BigBang {


template<typename dtype>
void Net<dtype>::Initialize() {
	const int layer_size = net_params_.layer_param_size();
	for (int i = 0; i < layer_size+1; ++i) {
		input_.push_back(std::make_shared<Tensor<dtype>>());
	}

	for (int i = 0; i < layer_size; ++i) {
		layers_.push_back(LayerRegistry<dtype>::CreateLayer(net_params_.layer_param(i)));
		layers_[i]->SetUp(input_[i].get(), input_[i + 1].get());
	}

	predicted_tensor_.reset(new Tensor<dtype>());
}

template<typename dtype>
void Net<dtype>::Train() {
	const int layer_size = layers_.size();
	if (layer_size == 0) return;

	layers_[0]->Forward(input_[0].get(), input_[1].get());
	//TODO: we can suppose that the index 0 layer save the label info now
	input_[input_.size() - 1]->shared_data(*(layers_[0]->Labels().get()));

	for (int i = 1; i < layer_size; ++i) {
		layers_[i]->Forward(input_[i].get(), input_[i+1].get());
	}

	for (int i = layer_size - 1; i >= 0; --i) {
		layers_[i]->Backward(input_[i+1].get(), input_[i].get());
	}
}

template<typename dtype>
int Net<dtype>::TestValidateData() {
	const int layer_size = layers_.size();
	for (int i = 0; i < layer_size; ++i) {
		layers_[i]->Forward(input_[i].get(), input_[i + 1].get());
	}
	auto output = input_[layer_size];
	predicted_tensor_->Reshape(std::vector<int>{output->shape(0), 1, 1, 1});
	predicted_tensor_->Reset();

	Tensor<int> count(std::vector<int>{1, 1, 1, 1});
	if (Config::Get().mode() == Config::ProcessUnit::CPU) {
		bigbang_cpu_argmax(output->cpu_data(), output->shape(0), output->shape(2)*output->shape(3),
			predicted_tensor_->mutable_cpu_data());
		bigbang_cpu_equals_count(layers_[0]->Labels().get()->cpu_data(), predicted_tensor_->cpu_data(),
			predicted_tensor_->size(), count.mutable_cpu_data());
	}
	else {
		bigbang_gpu_argmax(output->gpu_data(), output->shape(0), output->shape(2)*output->shape(3),
			predicted_tensor_->mutable_gpu_data());
		bigbang_gpu_equals_count(layers_[0]->Labels().get()->gpu_data(), predicted_tensor_->gpu_data(),
			predicted_tensor_->size(), count.mutable_gpu_data());
	}

	return count.cpu_data()[0];
}


INSTANTIATE_CLASS(Net);
}