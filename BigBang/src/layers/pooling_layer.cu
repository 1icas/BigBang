#include "../../include/layers/pooling_layer.h"
//sc
#include <iostream>
//
#include <device_launch_parameters.h>

#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/util/common.h"

template<typename dtype>
__global__ void MaxPool(const int count, const dtype* input, const int channels,  
	const int height, const int width,  const int pool_h, const int pool_w, 
	const int pool_stride_h, const int pool_stride_w, int* pool_pos, dtype* output) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		const int output_h = (height - pool_h) / pool_stride_h + 1;
		const int output_w = (width - pool_w) / pool_stride_w + 1;
		const int s = output_h * output_w;
		const int c = s * channels;
		const int i = index / c;
		const int channels_i = (index - c*i) / s;
		const int h_i = (index - c*i - s*channels_i) / output_w;
		const int w_i = index - c*i - s*channels_i - h_i * output_w;
		const int h = h_i * pool_stride_h * width;
		const int w = w_i * pool_stride_w;
		output[index] = input[i * height * width * channels + channels_i *height * width + h + w];
		pool_pos[index] = 0;
		for (int pool_row = 0; pool_row < pool_h; ++pool_row) {
			for (int pool_column = 0; pool_column < pool_w; ++pool_column) {
				int d = i * height * width * channels + channels_i *height * width + h + w + pool_row*width + pool_column;
				if (output[index] < input[d]) {
					pool_pos[index] = pool_row*pool_w + pool_column;
					output[index] = input[d];
				}
			}
		}

	}
}

template <typename dtype>
__global__ void MaxPoolBackward(const int count, const dtype* input, const int input_channels, const int input_h, 
	const int input_w, const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, 
	const int* pool_pos, const int output_h, const int output_w, dtype* output) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < count) {
		const int s = input_h * input_w;
		const int c = s * input_channels;
		const int i = index / c;
		const int channels_i = (index - c*i) / s;
		const int h_i = (index - c*i - s*channels_i) / input_w;
		const int w_i = index - c*i - s*channels_i - h_i * input_w;
		const int pos = pool_pos[index];
		output[i*input_channels*output_h*output_w + channels_i*output_h*output_w +
			(h_i*pool_stride_h + pos/pool_h)*output_w + (w_i*pool_stride_w + pos%pool_w)] = input[index];
	}
}

namespace BigBang {
	//ÔÝ²»¿¼ÂÇÖØµþ
	template<typename dtype>
	void PoolingLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		const dtype* bottom_data = bottom->gpu_data();
		dtype* top_data = top->mutable_gpu_data();

		switch (pool_) {
		case PoolingLayerParams<dtype>::Pool::MaxPool: {
			int* pos_data = max_pool_pos_->mutable_gpu_data();
			const int size = top->size();
			MaxPool<<<BigBangGetBlocks(size),THREAD_MAX_NUMS >>>(size, bottom_data, bottom_channels_, 
				bottom_row_, bottom_column_, pool_h_, pool_w_, stride_h_, stride_w_, pos_data,
				top_data);
		}
		break;

		default:
			std::cout << "only support max pool now" << std::endl;
			THROW_EXCEPTION;
			break;
		}

		//sc
		/*const dtype* top_data_cpu = top->cpu_data();
		for (int i = 0; i < top->size(); ++i) {
			std::cout << top_data_cpu[i] << std::endl;
		}*/
		//

	}

	template<typename dtype>
	void PoolingLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
		const dtype* top_data = top->gpu_data();
		const dtype* top_diff_data = top->gpu_diff_data();
		dtype* bottom_diff_data = bottom->mutable_gpu_diff_data();
		bigbanggpumemset(bottom_diff_data, 0, sizeof(dtype)*bottom->size());
		switch (pool_) {
		case PoolingLayerParams<dtype>::Pool::MaxPool: {
			const int size = top->size();
			MaxPoolBackward<<<BigBangGetBlocks(size), THREAD_MAX_NUMS >>>(size, top_diff_data, top_channels_, top_row_, top_column_,
				pool_h_, pool_w_, stride_h_, stride_w_, max_pool_pos_->gpu_data(), bottom_row_,
				bottom_column_, bottom_diff_data);

		}
																									 break;

		default:
			std::cout << "only support max pool now" << std::endl;
			THROW_EXCEPTION;
			break;
		}

		//sc
		/*for (int i = 0; i < bottom->size(); ++i) {
			std::cout << bottom->cpu_diff_data()[i] << std::endl;
		}*/
		//

	}

	INSTANTIATE_CLASS_GPU_FUNCTION(PoolingLayer);
}