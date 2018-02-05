#include "../../include/layers/dropout_layer.h"
#include <random>
#include "../../include/layer_factory.h"

namespace BigBang {

template<typename dtype>
void DropoutLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	auto shape = bottom->shape();
	auto ratio = dropout_params_.dropout_ratio();
	top->Reshape(shape);
	mask_.reset(new Tensor<unsigned int>(shape));
	scale_ = 1. / (1. - ratio);
	threshold_ = static_cast<unsigned int>(UINT_MAX * ratio);
}

template<typename dtype>
void DropoutLayer<dtype>::reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	auto shape = bottom->shape();
	top->Reshape(shape);
	mask_->Reshape(shape);
}

template<typename dtype>
void DropoutLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const int size = bottom->size();
	const dtype* bottom_data = bottom->cpu_data();
	std::bernoulli_distribution bd(1.-dropout_params_.dropout_ratio());
	std::mt19937 mt;
	dtype* mutable_top_data = top->mutable_cpu_data();
	unsigned int* mask_data = mask_->mutable_cpu_data();
	for (int i = 0; i < size; ++i) {
		mask_data[i] = bd(mt);
		mutable_top_data[i] = bottom_data[i] * mask_data[i] * scale_;
	}
}

template<typename dtype>
void DropoutLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = bottom->size();
	const dtype* top_diff_data = top->cpu_diff_data();
	const unsigned int* mask_data = mask_->cpu_data();
	dtype* bottom_diff_data = bottom->mutable_cpu_diff_data();
	for (int i = 0; i < size; ++i) {
		bottom_diff_data[i] = top_diff_data[i] * mask_data[i] * scale_;
	}
}

INSTANTIATE_CLASS(DropoutLayer);
REGISTRY_LAYER(Dropout);
}