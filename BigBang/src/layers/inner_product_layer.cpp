#include "../../include/layers/inner_product_layer.h"

#include <cassert>

#include "../../include/base.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"
#include "../../include/layers/layer_type_macro.h"
#include "../../include/util/common.h"
#include "../../include/util/math_function_ptr.h"

#include <iostream>

namespace BigBang {

template<typename dtype>
void InnerProductLayer<dtype>::SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	VALIDATE_POINTER(bottom);
	VALIDATE_POINTER(top);
	VALIDATE_POINTER(weights_.get());
	if (use_biases_) VALIDATE_POINTER(biases_.get());
	Prepare(bottom, top);
	Check(bottom, top);
}

// we suppose all of the Tensor is a four dimension data now
// so we should be see the bottom shape(0) dimension as a row
// and the other dimension(shape(1), shape(2), shape(3)) as a column
template<typename dtype>
void InnerProductLayer<dtype>::Prepare(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	CHECK_EQ(top->dimension(), DATA_DIMENSION);
	CHECK_EQ(weights_->dimension(), PARAMS_DIMENSION);
	bottom_row_ = bottom->shape(0);
	bottom_column_ = bottom->shape(1);
	for (int i = 2; i < bottom->dimension(); ++i) {
		bottom_column_ *= bottom->shape(i);
	}
	weights_row_ = weights_->shape(2);
	weights_column_ = weights_->shape(3);
	top_row_ = top->shape(0);
	top_column_ = top->shape(3);
	if (use_biases_) {
		CHECK_EQ(biases_->dimension(), PARAMS_DIMENSION);
		biases_row_ = biases_->shape(2);
		biases_column_ = biases_->shape(3);
		middle_ = std::make_shared<Tensor<dtype>>(std::vector<int>{1, 1, top_row_, 1});
		dtype* middle_mutable_data = middle_->mutable_cpu_data();
		for (int i = 0; i < middle_->size(); ++i) {
			middle_mutable_data[i] = static_cast<dtype>(1);
		}
	}
}

template<typename dtype>
void InnerProductLayer<dtype>::Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {
	CHECK_EQ(bottom_row_, top_row_);
	CHECK_EQ(bottom_column_, weights_row_);
	CHECK_EQ(top_column_, weights_column_);
	if (use_biases_) {
		CHECK_EQ(biases_row_, weights_column_);
		CHECK_EQ(biases_column_, 1);
	}
}


template <typename dtype>
void InnerProductLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	//bigbangcpumemset(top->mutable_cpu_data(), 0, sizeof(dtype)*top->size());

	/*bigbang_cpu_gemm(bottom->cpu_data(), bottom_row_, bottom_column_, false, weights_->cpu_data(), 
		weights_row_, weights_column_, false, 1.0, use_biases_ ? biases_->cpu_data() : (dtype*)nullptr, 
		biases_row_, biases_column_, false, top->mutable_cpu_data());*/
	bigbang_cpu_gemm<dtype>(false, false, bottom_row_, weights_column_, bottom_column_, 1.,
		bottom->cpu_data(), weights_->cpu_data(), 0., top->mutable_cpu_data());

	if (use_biases_) {
		bigbang_cpu_gemm<dtype>(false, false, top_row_, top_column_, 1, 1.,
			middle_->cpu_data(), biases_->cpu_data(), 1., top->mutable_cpu_data());
	}
}

//top -> n*10
//bottom -> n*30
//weights -> 30 * 10
template <typename dtype>	
void InnerProductLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* bottom_data = bottom->cpu_data();
	const dtype* top_diff_data = top->cpu_diff_data();
	//bigbangcpumemset(bottom->mutable_cpu_diff_data(), 0, sizeof(dtype)*bottom->size());

	//get the delta
	/*bigbang_cpu_gemm(top_diff_data, top_row_, top_column_, false, weights_->cpu_data(), weights_row_,
		weights_column_, true, 1.0, (dtype*)nullptr, 0, 0, false, bottom->mutable_cpu_diff_data());*/
	bigbang_cpu_gemm<dtype>(false, true, top_row_, weights_row_, top_column_, 1., top_diff_data,
		weights_->cpu_data(), 0, bottom->mutable_cpu_diff_data());

	UpdateParams_CPU(bottom_data, top_diff_data);
}

//input is z (sigmod(x*w + b))
template <typename dtype>
void InnerProductLayer<dtype>::UpdateParams_CPU(const dtype* bottom_data, const dtype* delta) {
	//update the biases
	if (use_biases_) {
		dtype* biases_mutable_diff_data = biases_->mutable_cpu_diff_data();
		bigbangcpumemset(biases_mutable_diff_data, 0, sizeof(dtype)*biases_->size());
		bigbang_cpu_column_sum_plus(delta, bottom_row_, biases_row_, biases_mutable_diff_data);
		bigbang_cpu_minus(biases_->cpu_data(), biases_mutable_diff_data, biases_row_, alpha_ / bottom_row_, 
			biases_->mutable_cpu_data());
	}

	//update the weights
	dtype* weights_diff_data = weights_->mutable_cpu_diff_data();
	//bigbangcpumemset(weights_diff_data, 0, sizeof(dtype)*weights_->size());
	/*bigbang_cpu_gemm(bottom_data, bottom_row_, bottom_column_, true, delta, top_row_, top_column_,
		false, alpha_ / bottom_row_, (dtype*)nullptr, 0, 0, false, weights_diff_data);*/
	bigbang_cpu_gemm<dtype>(true, false, bottom_column_, top_column_, bottom_row_, alpha_ / bottom_row_,
		bottom_data, delta, 0, weights_diff_data);
	bigbang_cpu_minus(weights_->cpu_data(), weights_diff_data, weights_row_*weights_column_, static_cast<dtype>(1.0), weights_->mutable_cpu_data());
}

#ifndef CPU_ONLY

template<typename dtype>
void InnerProductLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	bigbang_gpu_gemm<dtype>(false, false, bottom_row_, weights_column_, bottom_column_, 1.,
		bottom->gpu_data(), weights_->gpu_data(), 0., top->mutable_gpu_data());
	//sc
	/*for (int i = 0; i < top->size(); ++i) {
		std::cout << top->cpu_data()[i] << std::endl;
	}*/
	//
	if (use_biases_) {
		bigbang_gpu_gemm<dtype>(false, false, top_row_, top_column_, 1, 1.,
			middle_->gpu_data(), biases_->gpu_data(), 1., top->mutable_gpu_data());
	}

}

template<typename dtype>
void InnerProductLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const dtype* bottom_data = bottom->gpu_data();
	const dtype* top_diff_data = top->gpu_diff_data();
	//get the delta
	bigbang_gpu_gemm<dtype>(false, true, top_row_, weights_row_, top_column_, 1., top_diff_data,
		weights_->gpu_data(), 0, bottom->mutable_gpu_diff_data());
	UpdateParams_GPU(bottom_data, top_diff_data);
	//sc
	/*for (int i = 0; i < top->size(); ++i) {
		std::cout << top->cpu_data()[i] << std::endl;
	}*/
	//
}

template<typename dtype>
void InnerProductLayer<dtype>::UpdateParams_GPU(const dtype* bottom_data, const dtype* delta) {
	//update the biases
	if (use_biases_) {
		dtype* biases_mutable_diff_data = biases_->mutable_gpu_diff_data();
		bigbang_gpu_column_sum_plus(delta, bottom_row_, biases_row_, biases_mutable_diff_data);
		bigbang_gpu_minus(biases_->gpu_data(), biases_mutable_diff_data, biases_row_, alpha_ / bottom_row_,
			biases_->mutable_gpu_data());
	}

	//update the weights
	dtype* weights_diff_data = weights_->mutable_gpu_diff_data();
	bigbang_gpu_gemm<dtype>(true, false, bottom_column_, top_column_, bottom_row_, alpha_ / bottom_row_,
		bottom_data, delta, 0, weights_diff_data);
	bigbang_gpu_minus(weights_->gpu_data(), weights_diff_data, weights_row_*weights_column_,
		static_cast<dtype>(1.0), weights_->mutable_gpu_data());

	//sc
	/*dtype* weights_diff_data_cpu = weights_->mutable_cpu_diff_data();
	for (int i = 0; i < weights_->size(); ++i) {
		std::cout << weights_diff_data_cpu[i] << std::endl;
	}*/
	//

}

#endif

INSTANTIATE_CLASS(InnerProductLayer);
REGISTRY_LAYER(InnerProduct);
}

