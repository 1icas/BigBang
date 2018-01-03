#include "../../include/layers/softmax_cost_layer.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "../../include/util/common.h"

template<typename dtype>
__global__ void gpu_find_max_value(const dtype* input, const int num, const int single_size, dtype* output) {
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
__global__ void gpu_exp(const dtype* a, const int size, const int per_data_size, const dtype* max, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		const int pos = index / per_data_size;
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
__global__ void gpu_exp_div(const dtype* a, const int size, const int per_data_size, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		const int pos = index / per_data_size;
		b[index] /= a[pos];
	}
}

template<typename dtype>
__global__ void log_likelihoold_derivation(const dtype* a, const int size, const int per_data_size, const dtype* labels,	
	dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		const int pos = index / per_data_size;
		const int cur_pos = index % per_data_size;
		const int label = labels[pos];
		if (cur_pos == label) {
			b[index] = a[index] - 1;
		}
		else {
			b[index] = a[index];
		}
	}
}

//template<typename dtype>
//__global__ void softmax_cost(const dtype* result, const int size, const int num, const dtype* labels, dtype* loss) {
//	const int index = blockIdx.x * blockDim.x + threadIdx.x;
//	if (index < size) {
//		const int pos = index / num;
//		const int cur_pos = index % num;
//		const int label = labels[pos];
//		if (cur_pos == label) {
//			dtype c = -log(std::max<dtype>(result[index], FLT_MIN));
//			atomicAdd((double*)loss, (double)c);
//		}
//	}
//}

namespace BigBang {

template<typename dtype>
void SoftmaxCostLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const dtype* bottom_data = bottom->gpu_data();
	const int nums = bottom->shape(0);
	const int size = bottom->size();
	const int per_data_size = size / nums;
	//TODO: can we use another ways to do this thing(or Can we don't need memset the memory ?)
	softmax_sum_->Reset();
	dtype* softmax_sum_data = softmax_sum_->mutable_gpu_data();
	dtype* softmax_result_data = softmax_result_->mutable_gpu_data();
	dtype* mutable_max_gpu_data = max_num_->mutable_gpu_data();
	gpu_find_max_value<<<BigBangGetBlocks(nums), THREAD_MAX_NUMS >>>(bottom_data, nums, per_data_size, mutable_max_gpu_data);
	gpu_exp<<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>>(bottom_data, size, per_data_size, mutable_max_gpu_data, softmax_result_data);
	gpu_exp_sum <<<BigBangGetBlocks(nums), THREAD_MAX_NUMS >>> (softmax_result_data, nums,
		per_data_size, softmax_sum_data);
	gpu_exp_div <<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>> (softmax_sum_data, size, per_data_size, softmax_result_data);

	//compute the error
	if (++count_ % times_ == 0) {
		const dtype* labels = top->cpu_data();
		const dtype* softmax_result_cpu_data = softmax_result_->cpu_data();
		dtype loss = 0;
		for (int i = 0; i < nums; ++i) {
			loss += -log(softmax_result_cpu_data[static_cast<int>(labels[i] + 0.1) + i*per_data_size]);
		}
		std::cout << "training " << count_++ << " times, the error is: " << loss / nums << std::endl;
	}
}

template<typename dtype>
void SoftmaxCostLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = bottom->size();
	const int nums = bottom->shape(0);
	const dtype* top_data = top->gpu_data();
	const dtype* result_data = softmax_result_->gpu_data();
	dtype* bottom_diff = bottom->mutable_gpu_diff_data();
	log_likelihoold_derivation <<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>> (result_data, 
		size, size / nums, top_data, bottom_diff);

	//for (int i = 0; i < bottom->size(); ++i) {
	//	std::cout << bottom->cpu_diff_data()[i] << std::endl;
	//}

}
INSTANTIATE_CLASS_GPU_FUNCTION(SoftmaxCostLayer);

}