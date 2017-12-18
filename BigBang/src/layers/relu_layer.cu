#include "../../include/layers/relu_layer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/util/common.h"

template<typename dtype>
__global__ void relu_forward(const int size, const dtype* bottom_data, dtype* top_data) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		top_data[index] = bottom_data[index] > 0 ? bottom_data[index] : 0;
	}
}

template<typename dtype>
__global__ void relu_backward(const int size, const dtype* bottom_data, const dtype* top_diff_data,
	dtype* bottom_diff_data) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		bottom_diff_data[index] = bottom_data[index] > 0 ? top_diff_data[index] * 1 : 0;
	}
}

namespace BigBang {
	template<typename dtype>
	void ReluLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		const int size = bottom->size();
		const dtype* bottom_data = bottom->gpu_data();
		dtype* top_data = top->mutable_gpu_data();
		relu_forward << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> > (size,
			bottom_data, top_data);

	}

	template<typename dtype>
	void ReluLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
		const int size = bottom->size();
		const dtype* bottom_data = bottom->gpu_data();
		const dtype* top_diff_data = top->gpu_diff_data();
		dtype* bottom_diff_data = bottom->mutable_gpu_diff_data();
		relu_backward << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> >(size, bottom_data, top_diff_data, bottom_diff_data);
	}

	INSTANTIATE_CLASS_GPU_FUNCTION(ReluLayer);



}