#include "../../include/layers/mse_layer.h"

#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"
#include "../../include/util/math_function_ptr.h"

#include <iostream>

namespace BigBang {

template<typename dtype>
void MSELayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	//top->Reshape(bottom->shape());
	top->Reshape(std::vector<int>{bottom->shape(0), 1, 1, 1});
	result_.reset(new Tensor<dtype>(bottom->shape()));
}

//template<typename dtype>
//void MSELayer<dtype>::Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
//	CHECK_EQ(top->size(), bottom->size());
//}

template<typename dtype>
void MSELayer<dtype>::Forward_CPU(const Tensor<dtype>* input, Tensor<dtype>* output) {
	//double m = 0.0;
	//const int row = input->shape(0);
	//const int column = input->shape(3);
	//const dtype* r_data = output->cpu_data();
	//const dtype* i_data = input->cpu_data();
	//result_->Reset();
	//dtype* result_data = result_->mutable_cpu_data();
	//const int num = result_->shape(0);
	//const int classes = result_->shape(3);
	//for (int i = 0; i < num; ++i) {
	//	//TODO£º
	//	result_data[i*classes + static_cast<int>(r_data[i] + 0.1)] = 1;
	//}
	//for (int i = 0; i < row; ++i) {
	//	for (int k = 0; k < column; ++k) {
	//		m += pow((result_data[i*column + k] - i_data[i*column + k]), 2);
	//	}
	//}

	//std::cout << "the cost function is: " << m / row << std::endl;
}

//suppose all of the label index start at zero
template<typename dtype>
void MSELayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* bottom_data = bottom->cpu_data();
	const dtype* top_data = top->cpu_data();
	const int num = result_->shape(0);
	const int classes = result_->shape(3);
	dtype* diff_data = bottom->mutable_cpu_diff_data();
	result_->Reset();
	dtype* result_data = result_->mutable_cpu_data();
	bigbang_cpu_gen_fit_label(top_data, num, classes, result_data);
	bigbang_cpu_minus(bottom_data, result_data, bottom->size(), static_cast<dtype>(1.0), diff_data);
	//for (int i = 0; i < bottom->size(); ++i) {
	//	std::cout << diff_data[i] << std::endl;
	//}
}

INSTANTIATE_CLASS(MSELayer);
REGISTRY_LAYER(MSE);
}
