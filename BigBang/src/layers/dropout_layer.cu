#include "../../include/layers/dropout_layer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/util/math_function_ptr.h"

template<typename dtype>
__global__ void dropout_forward_backward(const unsigned int threshold, const dtype scale, const int size, 
	const unsigned int* mask, const dtype* input, dtype* output) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		output[index] = input[index] * (mask[index] > threshold) * scale;
	}
}


namespace BigBang {

template<typename dtype>
void DropoutLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const int size = bottom->size();
	unsigned int* mask_data = mask_->mutable_gpu_data();
	bigbang_gpu_random_uniform(size, mask_data);
	const dtype* bottom_data = bottom->gpu_data();
	dtype* top_data = top->mutable_gpu_data();
	dropout_forward_backward<<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>>(threshold_, scale_, size, mask_data, bottom_data, top_data);
}

template<typename dtype>
void DropoutLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = top->size();
	unsigned int* mask_data = mask_->mutable_gpu_data();
	const dtype* top_diff_data = top->gpu_diff_data();
	dtype* bottom_diff_data = bottom->mutable_gpu_diff_data();
	dropout_forward_backward<<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>>(threshold_, scale_, size, mask_data, top_diff_data, bottom_diff_data);
}

INSTANTIATE_CLASS_GPU_FUNCTION(DropoutLayer);
}