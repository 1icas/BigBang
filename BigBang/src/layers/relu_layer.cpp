#include "../../include/layers/relu_layer.h"
#include <algorithm>
#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"


namespace BigBang {

	template<typename dtype>
	void ReluLayer<dtype>::SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
		VALIDATE_POINTER(bottom);
		VALIDATE_POINTER(top);
	}

	template<typename dtype>
	void ReluLayer<dtype>::Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
		CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
		CHECK_EQ(top->dimension(), DATA_DIMENSION);
		CHECK_EQ(bottom, top);
	}

	template<typename dtype>
	void ReluLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		const int size = bottom->size();
		const dtype* bottom_data = bottom->cpu_data();
		dtype* top_data = top->mutable_cpu_data();
		for (int i = 0; i < size; ++i) {
			top_data[i] = std::max(bottom_data[i], static_cast<dtype>(0));
		}
	}

	template<typename dtype>
	void ReluLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
		const int size = bottom->size();
		const dtype* top_diff_data = top->cpu_diff_data();
		const dtype* bottom_data = bottom->cpu_data();
		dtype* bottom_diff_data = bottom->mutable_cpu_diff_data();
		for (int i = 0; i < size; ++i) {
			bottom_diff_data[i] =  (bottom_data[i] > 0 ? top_diff_data[i] * bottom_data[i] : 0);
		}
	}

	INSTANTIATE_CLASS(ReluLayer);
	REGISTRY_LAYER(Relu);
}
