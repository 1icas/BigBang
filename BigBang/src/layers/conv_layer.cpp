#include "../../include/layers/conv_layer.h"

#include <memory>
#include <vector>

#include "../../include/base.h"
#include "../../include/layer_factory.h"
#include "../../include/util/common.h"
#include "../../include/util/im2col.h"
#include "../../include/util/math_function_ptr.h"

namespace BigBang {

template<typename dtype>
void ConvLayer<dtype>::SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	VALIDATE_POINTER(bottom);
	VALIDATE_POINTER(top);
	Prepare(bottom, top);
	Check(bottom, top);
}

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

	const int output_h = (bottom_row_ - kernel_h_ + 2 * padding_h_) / stride_h_ + 1;
	const int output_w = (bottom_column_ - kernel_w_ + 2 * padding_w_) / stride_w_ + 1;
	CHECK_EQ(top_row_, output_h);
	CHECK_EQ(top_column_, output_w);

	if (use_biases_) {
		CHECK_EQ(biases_groups_, kernel_groups_);
		CHECK_EQ(biases_channels_, kernel_channels_);
		CHECK_EQ(biases_->shape(2), 1);
		CHECK_EQ(biases_->shape(3), 1);
	}
}

template<typename dtype>
void ConvLayer<dtype>::Prepare(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	CHECK_EQ(top->dimension(), DATA_DIMENSION);
	nums_ = bottom->shape(0);
	bottom_channels_ = bottom->shape(1);
	bottom_row_ = bottom->shape(2);
	bottom_column_ = bottom->shape(3);
	top_channels_ = top->shape(1);
	top_row_ = top->shape(2);
	top_column_ = top->shape(3);
	if (use_biases_) {
		CHECK_EQ(biases_->dimension(), PARAMS_DIMENSION);
		biases_groups_ = biases_->shape(0);
		biases_channels_ = biases_->shape(1);
	}
	const int unroll_h = bottom_channels_ * kernel_h_ * kernel_w_;
	const int unroll_w = top_row_ * top_column_;
	unroll_bottom_ = std::make_shared<Tensor<dtype>>(std::vector<int>{nums_, 1, unroll_h, unroll_w});
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
		im2col(bottom_data + i * bottom_offset, bottom_channels_, bottom_row_, bottom_column_, kernel_h_, kernel_w_, 
			padding_h_, padding_w_, stride_h_, stride_w_, dilation_h_, dilation_w_, unroll_bottom_data + i * unroll_bottom_offset);
		bigbang_cpu_gemm(kernels_data, kernel_groups_, kernel_channels_ * kernel_h_ * kernel_w_, false,
			unroll_bottom_data + i * unroll_bottom_offset, unroll_bottom_row, unroll_bottom_column, false, 1.0,
			(dtype*)nullptr, 0, 0, false, top_data + i*top_channels_*top_row_*top_column_);
	}

	if (use_biases_) {
		dtype* biases_data = biases_->mutable_cpu_data();
		for (int i = 0; i < nums_; ++i) {
			for (int k = 0; k < biases_groups_; ++k) {
				dtype biases = 0;
				for (int j = 0; j < biases_channels_; ++j) {
					biases += biases_data[k*biases_channels_ + j];
				}
				const int offset = i*biases_groups_*top_row_*top_column_ + k*top_row_*top_column_;
				plus(top_data + offset, top_row_*top_column_, biases, top_data + offset);
			}
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
		const int temp_size = kernel_channels_*kernel_h_*kernel_w_*top_row_*top_column_;
		dtype* temp = new dtype[temp_size];
		bigbangmemset(temp, 0, sizeof(dtype) * temp_size);
		bigbang_cpu_gemm(kernels_->mutable_cpu_data(), kernel_groups_, kernel_channels_*kernel_h_*kernel_w_, true,
			top_diff_data + i*top_channels_*top_row_*top_column_, top_channels_, top_row_*top_column_, false, 1.0, 
			(dtype*)(nullptr), 0, 0, false, temp);
		col2im(temp, bottom_channels_, bottom_row_, bottom_column_, kernel_h_, kernel_w_, padding_h_, padding_w_,
			stride_h_, stride_w_, 0, 0, bottom_mutable_diff_data + i*bottom_channels_*bottom_row_*bottom_column_);
		delete[]temp;
	}

	const dtype* bottom_data = bottom->cpu_data();
	UpdateParams(bottom_data, top_diff_data);
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
void ConvLayer<dtype>::UpdateParams(const dtype* bottom_data, const dtype* delta) {
	const int bottom_unroll_row = unroll_bottom_->shape(2);
	const int bottom_unroll_column = unroll_bottom_->shape(3);
	//update biases_
	if (use_biases_) {
		const int biases_size = biases_->size();
		dtype* biases_diff_data = biases_->mutable_cpu_diff_data();
		for (int i = 0; i < nums_; ++i) {
			row_sum_plus(delta + i*top_channels_*top_row_*top_column_, top_channels_, top_row_*top_column_, biases_diff_data);
		}
		dtype* biases_mutable_data = biases_->mutable_cpu_data();
		bigbang_cpu_minus(biases_mutable_data, biases_diff_data, biases_size, alpha_ / nums_, biases_mutable_data);
	}

	//update kernels
	const int kernel_size = kernels_->size();
	dtype* mutable_kernels_diff_data = kernels_->mutable_cpu_diff_data();
	bigbangmemset(mutable_kernels_diff_data, 0, sizeof(dtype)*kernel_size);
	const int unroll_size = bottom_unroll_row*bottom_unroll_column;
	for (int i = 0; i < nums_; ++i) {
		bigbang_cpu_gemm(delta + i*top_channels_*top_row_*top_column_, top_channels_, top_row_*top_column_, false, 
			bottom_data + unroll_size*i, bottom_unroll_row, bottom_unroll_column, true, 1.0, (dtype*)(nullptr),
			0, 0, false, mutable_kernels_diff_data);
	}
	dtype* kernel_data = kernels_->mutable_cpu_data();
	bigbang_cpu_minus(kernel_data, mutable_kernels_diff_data, kernel_size, alpha_ / nums_, kernel_data);
}

INSTANTIATE_CLASS(ConvLayer);
REGISTRY_LAYER(Conv);
}



