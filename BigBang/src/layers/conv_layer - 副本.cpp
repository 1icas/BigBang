#include "../../include/layers/conv_layer.h"

#include <memory>
#include <vector>

#include "../../include/base.h"
#include "../../include/filler.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"
#include "../../include/util/common.h"
#include "../../include/util/im2col.h"
#include "../../include/util/math_function_ptr.h"

#include <iostream>

namespace BigBang {

template<typename dtype>
void ConvLayer<dtype>::Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(bottom->shape(0), top->shape(0));
	CHECK_EQ(kernels_->dimension(), PARAMS_DIMENSION);
	CHECK_EQ(kernels_->shape(0), kernel_groups_);
	CHECK_EQ(kernels_->shape(1), kernel_channels_);
	CHECK_EQ(kernels_->shape(2), kernel_h_);
	CHECK_EQ(kernels_->shape(3), kernel_w_);
	CHECK_EQ(bottom_channels_, kernel_channels_);
	CHECK_EQ(top_channels_, kernel_groups_);
	
	if (use_bias_) {
		CHECK_EQ(biases_groups_, kernel_groups_);
		CHECK_EQ(biases_channels_, kernel_channels_);
		CHECK_EQ(biases_->shape(2), 1);
		CHECK_EQ(biases_->shape(3), 1);
	}
}

template<typename dtype>
void ConvLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	nums_ = bottom->shape(0);
	bottom_channels_ = bottom->shape(1);
	bottom_row_ = bottom->shape(2);
	bottom_column_ = bottom->shape(3);
	//generate the top tensor
	const int output_h = (bottom_row_ - kernel_h_ + 2 * padding_h_) / stride_h_ + 1;
	const int output_w = (bottom_column_ - kernel_w_ + 2 * padding_w_) / stride_w_ + 1;
	top->Reshape(std::vector<int>{nums_, kernel_groups_, output_h, output_w});
	top_channels_ = top->shape(1);
	top_row_ = top->shape(2);
	top_column_ = top->shape(3);
	kernels_ = std::make_shared<Tensor<dtype>>(std::vector<int>{kernel_groups_, kernel_channels_,
		kernel_h_, kernel_w_});
	CreateFiller<dtype>(conv_params_.kernel_filler())->Fill(kernels_.get());
	learnable_params_.push_back(kernels_);
	if (use_bias_) {
		biases_ = std::make_shared<Tensor<dtype>>(std::vector<int>{kernel_groups_, kernel_channels_, 1, 1});
		CreateFiller<dtype>(conv_params_.bias_filler())->Fill(biases_.get());
		biases_groups_ = biases_->shape(0);
		biases_channels_ = biases_->shape(1);
		middle_ = std::make_shared<Tensor<dtype>>(std::vector<int>{1, 1, biases_channels_, top_row_ * top_column_});
		dtype* middle_data = middle_->mutable_cpu_data();
		for (int i = 0; i < middle_->size(); ++i) {
			middle_data[i] = static_cast<dtype>(1);
		}
		learnable_params_.push_back(biases_);
	}
	const int unroll_h = bottom_channels_ * kernel_h_ * kernel_w_;
	const int unroll_w = top_row_ * top_column_;
	unroll_bottom_ = std::make_shared<Tensor<dtype>>(std::vector<int>{nums_, 1, unroll_h, unroll_w});
	relay_space_ = std::make_shared<Tensor<dtype>>(std::vector<int>{1, 1, 1, 
		kernel_channels_*kernel_h_*kernel_w_*top_row_*top_column_});
}

template <typename dtype>
void ConvLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const dtype* kernels_data = kernels_->cpu_data();
	const dtype* bottom_data = bottom->cpu_data();
	dtype* top_data = top->mutable_cpu_data();
	dtype* unroll_bottom_data = unroll_bottom_->mutable_cpu_data();
	const int bottom_offset = bottom_channels_ * bottom_row_ * bottom_column_;
	const int unroll_bottom_row = unroll_bottom_->shape(2);
	const int unroll_bottom_column = unroll_bottom_->shape(3);
	const int unroll_bottom_offset = unroll_bottom_row * unroll_bottom_column;

	for (int i = 0; i < nums_; ++i) {
		bigbang_cpu_im2col(bottom_data + i * bottom_offset, bottom_channels_, bottom_row_, bottom_column_, kernel_h_, kernel_w_,
			padding_h_, padding_w_, stride_h_, stride_w_, dilation_h_, dilation_w_, unroll_bottom_data + i * unroll_bottom_offset);
		bigbang_cpu_gemm(false, false, kernel_groups_, unroll_bottom_column, unroll_bottom_row, static_cast<dtype>(1.0), kernels_data,
			unroll_bottom_data + i * unroll_bottom_offset, static_cast<dtype>(0), top_data + i*top_channels_*top_row_*top_column_);
	}

	if (use_bias_) {
		/*dtype* biases_data = biases_->mutable_cpu_data();
		for (int i = 0; i < nums_; ++i) {
			for (int k = 0; k < biases_groups_; ++k) {
				dtype biases = 0;
				for (int j = 0; j < biases_channels_; ++j) {
					biases += biases_data[k*biases_channels_ + j];
				}
				const int offset = i*biases_groups_*top_row_*top_column_ + k*top_row_*top_column_;
				plus(top_data + offset, top_row_*top_column_, biases, top_data + offset);
			}
		}*/
		for (int i = 0; i < nums_; ++i) {
			bigbang_cpu_gemm(false, false, biases_groups_, top_row_*top_column_, biases_channels_, static_cast<dtype>(1.0), 
				biases_->cpu_data(), middle_->cpu_data(), static_cast<dtype>(1), top_data + i*top_channels_*top_row_*top_column_);
		}
		
	}
}

//for examples:
//feed forward:
//kernels-> 5*4
//input(unroll) -> 4*20
//output->5*20
//backward:
//output_delta->5*20 kernels->5*4
//new_delta kernels(T)*output_delta
//kernels_change output_delta*input(T)
template <typename dtype>
void ConvLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	//delta * w(T)

	//input  n*4*4
	//kernel 5*1*2*2  => 5*4
	//stride 1*1
	//unroll input n*4*9
	//output n*5*9 => n*5*3*3
	//
	//delta n*5*9
	//delta * kernel(T)

	const dtype* top_diff_data = top->cpu_diff_data();
	dtype* bottom_mutable_diff_data = bottom->mutable_cpu_diff_data();

	for (int i = 0; i < nums_; ++i) {
		dtype* temp = relay_space_->mutable_cpu_data();
		bigbang_cpu_gemm(true, false, kernel_channels_*kernel_h_*kernel_w_, top_row_*top_column_, kernel_groups_,
			static_cast<dtype>(1.0), kernels_->mutable_cpu_data(), top_diff_data + i*top_channels_*top_row_*top_column_,
			static_cast<dtype>(1.0), temp);
		bigbang_cpu_col2im(temp, bottom_channels_, bottom_row_, bottom_column_, kernel_h_, kernel_w_, padding_h_, padding_w_,
			stride_h_, stride_w_, 0, 0, bottom_mutable_diff_data + i*bottom_channels_*bottom_row_*bottom_column_);
	}

	const dtype* bottom_data = unroll_bottom_->cpu_data();
	UpdateParams_CPU(bottom_data, top_diff_data);
}

//for example
//input -> sigmoid(z(l-1)) 4*9
//weights -> x * 4
//delta -> 3*3

//output -> x * 9
//5*12
//5*9
//next delat *  12 * 9
//next delta * x(T)
template <typename dtype>
void ConvLayer<dtype>::UpdateParams_CPU(const dtype* bottom_data, const dtype* delta) {
	const int bottom_unroll_row = unroll_bottom_->shape(2);
	const int bottom_unroll_column = unroll_bottom_->shape(3);
	//update biases_
	if (use_bias_) {
		const int biases_size = biases_->size();
		dtype* biases_diff_data = biases_->mutable_cpu_diff_data();
		for (int i = 0; i < nums_; ++i) {
			row_sum_plus(delta + i*top_channels_*top_row_*top_column_, top_channels_, top_row_*top_column_, biases_diff_data);
		}
		dtype* biases_mutable_data = biases_->mutable_cpu_data();
	//	bigbang_cpu_minus(biases_mutable_data, biases_diff_data, biases_size, alpha_ / nums_, biases_mutable_data);
	}

	//update kernels
	const int kernel_size = kernels_->size();
	dtype* mutable_kernels_diff_data = kernels_->mutable_cpu_diff_data();
	//bigbangcpumemset(mutable_kernels_diff_data, 0, sizeof(dtype)*kernel_size);
	const int unroll_size = bottom_unroll_row*bottom_unroll_column;
	for (int i = 0; i < nums_; ++i) {
		bigbang_cpu_gemm(false, true, top_channels_, bottom_unroll_row, bottom_unroll_column, static_cast<dtype>(1.0),
			delta + i*top_channels_*top_row_*top_column_, bottom_data + unroll_size*i,
			i == 0 ? static_cast<dtype>(0.0) : static_cast<dtype>(1.0), mutable_kernels_diff_data);
	}

	dtype* kernel_data = kernels_->mutable_cpu_data();
	//bigbang_cpu_minus(kernel_data, mutable_kernels_diff_data, kernel_size, alpha_ / nums_, kernel_data);
}

template <typename dtype>
void ConvLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const dtype* kernels_data = kernels_->gpu_data();
	const dtype* bottom_data = bottom->gpu_data();
	dtype* top_data = top->mutable_gpu_data();
	dtype* unroll_bottom_data = unroll_bottom_->mutable_gpu_data();
	const int bottom_offset = bottom_channels_ * bottom_row_ * bottom_column_;
	const int unroll_bottom_row = unroll_bottom_->shape(2);
	const int unroll_bottom_column = unroll_bottom_->shape(3);
	const int unroll_bottom_offset = unroll_bottom_row * unroll_bottom_column;

	for (int i = 0; i < nums_; ++i) {
		bigbang_gpu_im2col(bottom_data + i * bottom_offset, bottom_channels_, bottom_row_, bottom_column_, kernel_h_, kernel_w_,
			padding_h_, padding_w_, stride_h_, stride_w_, dilation_h_, dilation_w_, unroll_bottom_data + i * unroll_bottom_offset);
		bigbang_gpu_gemm(false, false, kernel_groups_, unroll_bottom_column, unroll_bottom_row, static_cast<dtype>(1.0), kernels_data,
			unroll_bottom_data + i * unroll_bottom_offset, static_cast<dtype>(0), top_data + i*top_channels_*top_row_*top_column_);
	}

	//sc
	/*const dtype* top_data_cpu = top->cpu_data();
	for (int i = 0; i < top->size(); ++i) {
		std::cout << top_data_cpu[i] << std::endl;
	}*/
	//
	if (use_bias_) {
		for (int i = 0; i < nums_; ++i) {
			bigbang_cpu_gemm(false, false, biases_groups_, top_row_*top_column_, biases_channels_, static_cast<dtype>(1.0),
				biases_->gpu_data(), middle_->gpu_data(), static_cast<dtype>(1), top_data + i*top_channels_*top_row_*top_column_);
		}
	}
}

template <typename dtype>
void ConvLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* top_diff_data = top->gpu_diff_data();
	dtype* bottom_mutable_diff_data = bottom->mutable_gpu_diff_data();

	for (int i = 0; i < nums_; ++i) {
		//TODO: the gpu will memset the memory in the gemm
		dtype* temp = relay_space_->mutable_gpu_data();
		//bigbangcpumemset(temp, 0, sizeof(dtype)*relay_space_->size());
		bigbang_gpu_gemm(true, false, kernel_channels_*kernel_h_*kernel_w_, top_row_*top_column_, kernel_groups_,
			static_cast<dtype>(1.0), kernels_->mutable_gpu_data(), top_diff_data + i*top_channels_*top_row_*top_column_,
			static_cast<dtype>(1.0), temp);
		bigbang_gpu_col2im(temp, bottom_channels_, bottom_row_, bottom_column_, kernel_h_, kernel_w_, padding_h_, padding_w_,
			stride_h_, stride_w_, 0, 0, bottom_mutable_diff_data + i*bottom_channels_*bottom_row_*bottom_column_);
	}

	const dtype* bottom_data = unroll_bottom_->gpu_data();
	UpdateParams_GPU(bottom_data, top_diff_data);
}

template <typename dtype>
void ConvLayer<dtype>::UpdateParams_GPU(const dtype* bottom_data, const dtype* delta) {
	const int bottom_unroll_row = unroll_bottom_->shape(2);
	const int bottom_unroll_column = unroll_bottom_->shape(3);
	//update biases_
	/*if (use_bias_) {
		const int biases_size = biases_->size();
		dtype* biases_diff_data = biases_->mutable_cpu_diff_data();
		for (int i = 0; i < nums_; ++i) {
			row_sum_plus(delta + i*top_channels_*top_row_*top_column_, top_channels_, top_row_*top_column_, biases_diff_data);
		}
		dtype* biases_mutable_data = biases_->mutable_cpu_data();
		bigbang_cpu_minus(biases_mutable_data, biases_diff_data, biases_size, alpha_ / nums_, biases_mutable_data);
	}*/

	//update kernels
	const int kernel_size = kernels_->size();
	dtype* mutable_kernels_diff_data = kernels_->mutable_gpu_diff_data();
//	dtype* kernel_diff_data = kernel_diff_->mutable_gpu_data();
	//bigbanggpumemset(kernel_diff_data, 0, sizeof(dtype)*kernel_size);
	const int unroll_size = bottom_unroll_row*bottom_unroll_column;

	for (int i = 0; i < nums_; ++i) {
		bigbang_gpu_gemm(false, true, top_channels_, bottom_unroll_row, bottom_unroll_column, static_cast<dtype>(1.0),
			delta + i*top_channels_*top_row_*top_column_, bottom_data + unroll_size*i, 
			i == 0 ? static_cast<dtype>(0.0) : static_cast<dtype>(1.0), mutable_kernels_diff_data);
	}
	dtype* kernel_data = kernels_->mutable_gpu_data();
	bigbang_gpu_minus(kernel_data, mutable_kernels_diff_data, kernel_size, alpha_ / nums_, kernel_data);
	//sc
	/*for (int i = 0; i < kernel_size; ++i) {
		std::cout << kernels_->cpu_data()[i] << std::endl;
	}
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << std::endl;*/

	//
}



INSTANTIATE_CLASS(ConvLayer);
REGISTRY_LAYER(Conv);
}



