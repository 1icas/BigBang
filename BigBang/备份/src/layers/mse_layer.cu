#include "../../include/layers/mse_layer.h"
#include "../../include/util/math_function_ptr.h"

namespace BigBang {

template<typename dtype>
void MSELayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {

}

template<typename dtype>
void MSELayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* result_data = top->gpu_data();
	const dtype* bottom_data = bottom->gpu_data();
	dtype* diff_data = bottom->mutable_gpu_diff_data();
	bigbang_gpu_minus(bottom_data, result_data, top->size(), static_cast<dtype>(1.0), diff_data);
}


INSTANTIATE_CLASS_GPU_FUNCTION(MSELayer);

}