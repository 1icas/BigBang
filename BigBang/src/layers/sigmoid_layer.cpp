#include "../../include/layers/sigmoid_layer.h"

#include <cmath>

#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"


template<typename dtype>
inline dtype sigmoid(dtype d) {
	return 0.5 + 0.5 * tanh(0.5 * d);
}

namespace BigBang {

template<typename dtype>
void SigmoidLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	top->Reshape(bottom->shape());
}

template<typename dtype>
void SigmoidLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const int size = bottom->size();
	const dtype* bottom_data = bottom->cpu_data();
	dtype* top_data = top->mutable_cpu_data();
	for (int i = 0; i < bottom->size(); ++i) {
		top_data[i] = sigmoid(bottom_data[i]);
	}
}

template<typename dtype>
void SigmoidLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = bottom->size();
	const dtype* top_data = top->cpu_data();
	const dtype* top_diff_data = top->cpu_diff_data();
	const dtype v = static_cast<dtype>(1.);
	dtype* bottom_diff_data = bottom->mutable_cpu_diff_data();
	for (int i = 0; i < size; ++i) {
		bottom_diff_data[i] = top_diff_data[i] * top_data[i] * (v - top_data[i]);
	}
}

INSTANTIATE_CLASS(SigmoidLayer);
REGISTRY_LAYER(Sigmoid);
}

