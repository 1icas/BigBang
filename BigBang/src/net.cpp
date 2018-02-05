#include "../include/net.h"

#include "../include/base.h"
#include "../include/layer_factory.h"
#include "../include/layers/layer_type_macro.h"
#include "../include/util/math_function_ptr.h"

namespace BigBang {
template<typename dtype>
void Net<dtype>::Initialize() {
	const int layer_size = net_params_.layer_param_size();
	input_.push_back(std::make_shared<Tensor<dtype>>(std::vector<int>(4,1)));
	for (int i = 0; i < layer_size; ++i) {
		std::shared_ptr<Layer<dtype>> layer = LayerRegistry<dtype>::CreateLayer(net_params_.layer_param(i));
		input_.push_back(std::make_shared<Tensor<dtype>>());
		layers_.push_back(layer);
	}
	//train layer setup
	auto prev_train = input_[0];
	auto prev_test = prev_train;
	for (int i = 0; i < layers_.size(); ++i) {
		auto layer = layers_[i];
		if (layer->Phase() == LayerParameter::TRAIN) {
			layer->SetUp(prev_train.get(), input_[i + 1].get());
			prev_train = input_[i + 1];
		}
		else if (layer->Phase() == LayerParameter::TEST) {
			layer->SetUp(prev_test.get(), input_[i + 1].get());
			prev_test = input_[i + 1];
		}
		else {
			assert(i != 0);
			layer->SetUp(input_[i].get(), input_[i + 1].get());
			prev_train = prev_test = input_[i + 1];
		}
	} 
	predicted_tensor_.reset(new Tensor<dtype>());
}

template<typename dtype>
void Net<dtype>::Train() {
	const int layer_size = layers_.size();
	if (layer_size == 0) return;
	auto prev = input_[0];
	int last_train_layer_index = -1;
	int first_train_layer_index = -1;
	for (int i = 0; i < layer_size; ++i) {
		auto layer = layers_[i];
		if (layer->Phase() != LayerParameter::TEST) {
			if (first_train_layer_index == -1) first_train_layer_index = i;
			layer->Reshape(prev.get(), input_[i + 1].get());
			prev = input_[i + 1];
			last_train_layer_index = i;
		}
	}
	assert(last_train_layer_index != -1 && first_train_layer_index != -1);
	std::vector<std::shared_ptr<Tensor<dtype>>> back;
	layers_[first_train_layer_index]->Forward(input_[0].get(), input_[first_train_layer_index+1].get());
	input_[last_train_layer_index+1]->shared_data(*(layers_[first_train_layer_index]->Labels().get()));
	prev = input_[first_train_layer_index + 1];
	back.push_back(input_[0]);
	back.push_back(prev);
	for (int i = first_train_layer_index + 1; i < layer_size; ++i) {
		auto layer = layers_[i];
		if (layer->Phase() != LayerParameter::TEST) {
			layer->Forward(prev.get(), input_[i + 1].get());
			prev = input_[i + 1];
			back.push_back(prev);
		}
	}
	int flag = back.size() - 1;
	for (int i = layer_size - 1; i >= 0; --i) {
		auto layer = layers_[i];
		if (layer->Phase() != LayerParameter::TEST) {
			layer->Backward(back[flag].get(), back[flag-1].get());
			--flag;
		}
	}


	//layers_[0]->Forward(input_[0].get(), input_[1].get());
	////TODO: we can suppose that the index 0 layer save the label info now
	//input_[input_.size() - 2]->shared_data(*(layers_[0]->Labels().get()));

	//for (int i = 1; i < layer_size; ++i) {
	//	if(layers_[i]->Phase() != LayerParameter::TEST)
	//		layers_[i]->Forward(input_[i].get(), input_[i+1].get());
	//}

	//for (int i = layer_size - 1; i >= 0; --i) {
	//	if (layers_[i]->Phase() != LayerParameter::TEST)
	//		layers_[i]->Backward(input_[i+1].get(), input_[i].get());
	//}
}

template<typename dtype>
int Net<dtype>::Test() {

	const int layer_size = layers_.size();
	if (layer_size == 0) return 0;
	auto prev = input_[0];
	int first_test_layer_index = -1;
	int last_test_layer_index = -1;
	for (int i = 0; i < layer_size; ++i) {
		if (layers_[i]->Phase() != LayerParameter::TRAIN) {
			if (first_test_layer_index == -1) first_test_layer_index = i;
			layers_[i]->Reshape(prev.get(), input_[i + 1].get());
			prev = input_[i + 1];
			last_test_layer_index = i;
		}
	}
	assert(first_test_layer_index != -1 && last_test_layer_index != -1);

	prev = input_[first_test_layer_index + 1];
	layers_[first_test_layer_index]->Forward(input_[0].get(), prev.get());
	//input_[last_test_layer_index+1]->shared_data(*(layers_[first_test_layer_index]->Labels().get()));
	for (int i = first_test_layer_index + 1; i < layer_size; ++i) {
		if (layers_[i]->Phase() != LayerParameter::TRAIN) {
			layers_[i]->Forward(prev.get(), input_[i + 1].get());
			prev = input_[i + 1];
		}
	}
	auto predict = prev;
	predicted_tensor_->Reshape(std::vector<int>{predict->shape(0), 1, 1, 1});
	predicted_tensor_->Reset();

	Tensor<int> count(std::vector<int>{1, 1, 1, 1});
	if (Config::Get().mode() == Config::ProcessUnit::GPU) {
		bigbang_cpu_argmax(predict->cpu_data(), predict->shape(0), predict->shape(2)*predict->shape(3),
			predicted_tensor_->mutable_cpu_data());
		bigbang_cpu_equals_count(layers_[first_test_layer_index]->Labels().get()->cpu_data(), predicted_tensor_->cpu_data(),
			predicted_tensor_->size(), count.mutable_cpu_data());
	}
	else {
		bigbang_gpu_argmax(predict->gpu_data(), predict->shape(0), predict->shape(2)*predict->shape(3),
			predicted_tensor_->mutable_gpu_data());
		bigbang_gpu_equals_count(layers_[first_test_layer_index]->Labels().get()->gpu_data(), predicted_tensor_->gpu_data(),
			predicted_tensor_->size(), count.mutable_gpu_data());
	}

	return count.cpu_data()[0];
}

template<typename dtype>
void Net<dtype>::Run() {
	//const int layer_size = layers_.size();
	//if (layer_size == 0) return;

	//layers_[0]->Forward(input_[0].get(), input_[1].get());
	////TODO: we can suppose that the index 0 layer save the label info now
	//input_[input_.size() - 1]->shared_data(*(layers_[0]->Labels().get()));

	//for (int i = 1; i < layer_size; ++i) {
	//	layers_[i]->Forward(input_[i].get(), input_[i + 1].get());
	//}

	//for (int i = layer_size - 1; i >= 0; --i) {
	//	layers_[i]->Backward(input_[i + 1].get(), input_[i].get());
	//}
}


INSTANTIATE_CLASS(Net);
}