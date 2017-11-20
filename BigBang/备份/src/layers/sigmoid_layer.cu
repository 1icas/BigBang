#include "../../include/layers/sigmoid_layer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../include/gpu_config.h"

template<typename dtype>
__global__ void sigmoid(const int size, const dtype* in, dtype* out) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		out[index] = 0.5 + 0.5 * tanh(0.5 * in[index]);
	}
}

template<typename dtype>
__global__ void sigmoid_derivative(const int size, const dtype* in_diff,
	const dtype* in_data, dtype* out_diff) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		out_diff[index] = in_diff[index] * in_data[index] * (1. - in_data[index]);
	}
}

namespace BigBang {
template<typename dtype>
void SigmoidLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top){
	const int size = bottom->size();
	const dtype* bottom_data = bottom->gpu_data();
	dtype* top_data = top->mutable_gpu_data();
	sigmoid << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> > (size, 
		bottom_data, top_data);
	
}

template<typename dtype>
void SigmoidLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = bottom->size();
	const dtype* top_data = top->gpu_data();
	const dtype* top_diff_data = top->gpu_diff_data();
	dtype* bottom_diff_data = bottom->mutable_gpu_diff_data();
	sigmoid_derivative<<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>>(size, top_diff_data, top_data, bottom_diff_data);
}

INSTANTIATE_CLASS_GPU_FUNCTION(SigmoidLayer);



}