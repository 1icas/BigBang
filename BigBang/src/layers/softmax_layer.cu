#include "../../include/layers/softmax_layer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/util/common.h"

template<typename dtype>
__global__ void gpu_find_max_value(const dtype* input, const int num, const int single_size,dtype* output) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num) {
		const int pos = single_size * index;
		dtype max = input[pos];
		for (int i = 1; i < single_size; ++i) {
			if (input[pos + i] > max) {
				max = input[pos + i];
			}
		}
		output[index] = max;
	}
}

template<typename dtype>
__global__ void gpu_exp(const dtype* a, const int size, const int num, const dtype* max, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		const int pos = index / num;
		b[index] = exp(a[index] - max[pos]);
	}
}

template<typename dtype>
__global__ void gpu_exp_sum(const dtype* a, const int num, const int row, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < num) {
		for (int i = 0; i < row; ++i) {
			b[index] += a[index*row + i];
		}
	}
}

template<typename dtype>
__global__ void gpu_exp_div(const dtype* a, const int size, const int num, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		const int pos = index / num;
		b[index] /= a[pos];
	}
}

namespace BigBang {

	template<typename dtype>
	void SoftmaxLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		const dtype* bottom_data = bottom->gpu_data();
		const int nums = bottom->shape(0);
		const int size = bottom->size();
		const int per_data_size = size / nums;

		dtype* softmax_sum_data = softmax_sum_->mutable_gpu_data();
		dtype* top_data = top->mutable_gpu_data();
		dtype* mutable_max_num = max_num_->mutable_gpu_data();
		gpu_find_max_value << <BigBangGetBlocks(nums), THREAD_MAX_NUMS >> > (bottom_data, nums, per_data_size, mutable_max_num);
		gpu_exp << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> >(bottom_data, size, per_data_size, mutable_max_num, top_data);
		gpu_exp_sum << <BigBangGetBlocks(nums), THREAD_MAX_NUMS >> > (top_data, nums,
			per_data_size, softmax_sum_data);
		gpu_exp_div << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> > (softmax_sum_data, size, per_data_size, top_data);
	}

	INSTANTIATE_CLASS_GPU_FUNCTION(SoftmaxLayer);

}