#include "../../include/layers/relu_layer.h"
#include <algorithm>
#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"


namespace BigBang {


//TODO: also we can shared the bottom data to top data(use the same memory) (but we can later do it )
template<typename dtype>
void ReluLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	top->Reshape(bottom->shape());
}

template<typename dtype>
void ReluLayer<dtype>::reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	top->Reshape(bottom->shape());
}

template<typename dtype>
void ReluLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const int size = bottom->size();
	const dtype* bottom_data = bottom->cpu_data();
	const dtype v = static_cast<dtype>(0.);
	dtype* top_data = top->mutable_cpu_data();
	for (int i = 0; i < size; ++i) {
		top_data[i] = std::max(bottom_data[i], v);
		//std::cout << top_data[i] << std::endl;
	}
}

template<typename dtype>
void ReluLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = bottom->size();
	const dtype* top_diff_data = top->cpu_diff_data();
	const dtype* bottom_data = bottom->cpu_data();
	const dtype v = static_cast<dtype>(0.);
	dtype* bottom_diff_data = bottom->mutable_cpu_diff_data();
	for (int i = 0; i < size; ++i) {
		bottom_diff_data[i] =  (bottom_data[i] > v ? top_diff_data[i] : v);
	}
}

INSTANTIATE_CLASS(ReluLayer);
REGISTRY_LAYER(Relu);
}
