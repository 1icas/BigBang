#include "../../include/layers/mse_layer.h"

#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"
#include "../../include/util/math_function_ptr.h"

#include <iostream>

namespace BigBang {
	
template<typename dtype> 
void MSELayer<dtype>::SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	VALIDATE_POINTER(bottom);
	VALIDATE_POINTER(top);
	Check(bottom, top);
}

template<typename dtype>
void MSELayer<dtype>::Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	CHECK_EQ(top->dimension(), DATA_DIMENSION);
	CHECK_EQ(top->size(), bottom->size());
}

template<typename dtype>
void MSELayer<dtype>::Forward_CPU(const Tensor<dtype>* input, Tensor<dtype>* output) {
	//assert(input != nullptr && output != nullptr);
	//int r_row = input->shape(0);
	//int r_column = input->shape(1);
	/*int o_row = output->GetRow();
	int o_column = output->GetColumn();
	assert(r_row == o_row && r_column == o_column);
	*/
	double m = 0.0;
	const int row = input->shape(0);
	const int column = input->shape(3);
	const dtype* r_data = output->cpu_data();
	const dtype* i_data = input->cpu_data();
	for (int i = 0; i < row; ++i) {
		for (int k = 0; k < column; ++k) {
			m += pow((r_data[i*column + k] - i_data[i*column + k]), 2);
		}
	}

	std::cout << "the cost function is: " << m / row << std::endl;
}

template<typename dtype>
void MSELayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* bottom_data = bottom->cpu_data();
	const dtype* result_data = top->cpu_data();
	dtype* diff_data = bottom->mutable_cpu_diff_data();
	bigbang_cpu_minus(bottom_data, result_data, top->size(), static_cast<dtype>(1.0), diff_data);
	/*for (int i = 0; i < bottom->size(); ++i) {
		std::cout << diff_data[i] << std::endl;
	}*/
}

INSTANTIATE_CLASS(MSELayer);
REGISTRY_LAYER(MSE);
}
