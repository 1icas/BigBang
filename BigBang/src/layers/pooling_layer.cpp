#include "../../include/layers/pooling_layer.h"

#include <cassert>
#include <iostream>
#include <vector>

#include "../../include/layer_factory.h"

template<typename dtype>
void MaxPool(const dtype* input, const int channels, const int height, const int width,
	const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, int* pool_pos, dtype* output) {
	const int output_height = (height - pool_h) / pool_stride_h + 1;
	const int output_width = (width - pool_w) / pool_stride_w + 1;

	int* p = pool_pos;
	dtype* o = output;
	for (int channel_index = 0; channel_index < channels; ++channel_index) {
		for (int output_row = 0; output_row < output_height; ++output_row) {
			for (int output_column = 0; output_column < output_width; ++output_column) {
				int h = output_row * pool_stride_h * width;
				int w = output_column * pool_stride_w;
				*o = input[channel_index *height * width + h + w];
				*p = 0;
				for (int pool_row = 0; pool_row < pool_h; ++pool_row) {
					for (int pool_column = 0; pool_column < pool_w; ++pool_column) {
						int index = channel_index *height * width + h + w + pool_row*width + pool_column;
						if (*o < input[index]) {
							*p = pool_row*pool_w + pool_column;
							*o = input[index];
						}
					}
				}
				++p;
				++o;
			}
		}
	}
}

template void MaxPool<float>(const float* input, const int channels, const int height, const int width,
	const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, int* pool_pos, float* output);
template void MaxPool<double>(const double* input, const int channels, const int height, const int width,
	const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, int* pool_pos, double* output);

template <typename dtype>
void MaxPoolBackward(const dtype* input, const int channels, const int h, const int w,
	const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, const int* pool_pos, 
	const int output_h, const int output_w, dtype* output) {
	const int input_size = h*w;
	const int output_size = output_h*output_w;
	for (int i = 0; i < channels; ++i) {
		for (int row = 0; row < h; ++row) {
			for (int column = 0; column < w; ++column) {
				const int index = pool_pos[row*w + column];
 				output[i*output_size + (row*pool_stride_h + index/pool_h)*output_w + (column*pool_stride_w+index%pool_w)]
					= input[i*input_size + row*w + column];
			}
		}
	}
}

template void MaxPoolBackward<float>(const float* input, const int channels, const int h, const int w,
	const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, const int* pool_pos, 
	const int output_h, const int output_w, float* output);
template void MaxPoolBackward<double>(const double* input, const int channels, const int h, const int w,
	const int pool_h, const int pool_w, const int pool_stride_h, const int pool_stride_w, const int* pool_pos, 
	const int output_h, const int output_w, double* output);

namespace BigBang {

template<typename dtype>
void PoolingLayer<dtype>::SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	VALIDATE_POINTER(bottom);
	VALIDATE_POINTER(top);
	Prepare(bottom, top);
	Check(bottom, top);
}

template<typename dtype>
void PoolingLayer<dtype>::Prepare(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	CHECK_EQ(top->dimension(), DATA_DIMENSION);
	nums_ = bottom->shape(0);
	bottom_channels_ = bottom->shape(1);
	bottom_row_ = bottom->shape(2);
	bottom_column_ = bottom->shape(3);
	top_channels_ = top->shape(1);
	top_row_ = top->shape(2);
	top_column_ = top->shape(3);
	max_pool_pos_ = std::make_shared<Tensor<int>>(std::vector<int>{nums_,
		bottom_channels_, top_row_, top_column_});
}

template<typename dtype>
void PoolingLayer<dtype>::Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(top->shape(0), nums_);
	CHECK_EQ(bottom_channels_, top_channels_);
	const int h = (bottom_row_ - pool_h_) / stride_h_ + 1;;
	const int w = (bottom_column_ - pool_w_) / stride_w_ + 1;
	CHECK_EQ(top_row_, h);
	CHECK_EQ(top_column_, w);
}

//ÔÝ²»¿¼ÂÇÖØµþ
template<typename dtype>
void PoolingLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const dtype* bottom_data = bottom->cpu_data();
	dtype* top_data = top->mutable_cpu_data();
	switch (pool_) {
	case PoolingLayerParams<dtype>::Pool::MaxPool: {
		int* pos_data = max_pool_pos_->mutable_cpu_data();
		for (int i = 0; i < nums_; ++i) {
			MaxPool(bottom_data + i * bottom_channels_*bottom_row_*bottom_column_, bottom_channels_, bottom_row_,
				bottom_column_, pool_h_, pool_w_, stride_h_, stride_w_, pos_data + i*bottom_channels_*top_row_*top_column_,
				top_data + i*bottom_channels_*top_row_*top_column_);
		}
	}
																									break;

	default:
		std::cout << "only support max pool now" << std::endl;
		break;
	}

}

template<typename dtype>
void PoolingLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int top_per_size = top_row_*top_column_;
	const int bottom_per_size = bottom_row_*bottom_column_;
	const dtype* top_data = top->cpu_data();
	const dtype* top_diff_data = top->cpu_diff_data();
	dtype* bottom_diff_data = bottom->mutable_cpu_diff_data();

	switch (pool_) {
	case PoolingLayerParams<dtype>::Pool::MaxPool: {
		for (int i = 0; i < nums_; ++i) {
			MaxPoolBackward(top_diff_data + top_per_size*i, top_channels_, top_row_, top_column_, 
				pool_h_, pool_w_, stride_h_, stride_w_, 
				max_pool_pos_->cpu_data(), bottom_row_, bottom_column_, bottom_diff_data);
		}
	}
																									break;

	default:
		std::cout << "only support max pool now" << std::endl;
		break;
	}
}

INSTANTIATE_CLASS(PoolingLayer);
REGISTRY_LAYER(Pooling);
}
