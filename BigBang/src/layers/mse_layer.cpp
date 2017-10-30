#include "../../include/layers/mse_layer.h"

#include "../../include/base.h"
#include "../../include/layer_factory.h"
#include "../../include/math_function_ptr.h"

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
	CHECK_EQ(bottom, top);
	CHECK_EQ(correct_result_->size(), bottom->size());
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
	/*double m = 0.0;
	const dtype* r_data = result_->data();
	const dtype* i_data = input->data();
	for (int i = 0; i < r_row; ++i) {
		for (int k = 0; k < r_column; ++k) {
			m += pow((r_data[i*r_column + k] - i_data[i*r_column + k]), 2);
		}
	}

	std::cout << "the cost function is: " << m / r_row << std::endl;*/
}

//目前认为bottom应该是一个二维数组
template<typename dtype>
void MSELayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* top_data = top->cpu_data();
	const dtype* result_data = correct_result_->cpu_data();
	dtype* diff_data = bottom->mutable_cpu_diff_data();

	bigbang_cpu_minus(top_data, result_data, correct_result_->size(), static_cast<dtype>(1.0), diff_data);
}

INSTANTIATE_CLASS(MSELayer);
REGISTRY_LAYER(MSE);
}
