#ifndef LAYER_PARAMS_MANAGE_H
#define LAYER_PARAMS_MANAGE_H

#include <memory>
#include <string>

#include "tensor.h"

namespace BigBang {
enum class FillerParameters {
	NORMAL_DISTRIBUTION
};

template<typename dtype>
struct PoolingLayerParams {
	enum class Pool {
		MaxPool,
		AveragePool
	};

	PoolingLayerParams() = default;
	PoolingLayerParams(const Pool& pool, const int pool_h, const int pool_w, const int pool_stride_h,  const int pool_stride_w) :
		pool_(pool), pool_h_(pool_h), pool_w_(pool_w), pool_stride_h_(pool_stride_h), pool_stride_w_(pool_stride_w) {}

	Pool pool_ = Pool::MaxPool;
	int pool_h_ = 1;
	int pool_w_ = 1;
	int pool_stride_h_ = 1;
	int pool_stride_w_ = 1;
};

template<typename dtype>
struct ConvLayerParams {
	ConvLayerParams() = default;
	ConvLayerParams(const int kernel_groups, const int kernel_channels, const int kernel_h, 
		const int kernel_w, const int stride_h, const int stride_w, const int padding_h, const int padding_w, const dtype lambda, 
		const dtype alpha, bool use_biases, const FillerParameters& weights_filler, 
		const FillerParameters& biases_filler) : kernel_groups_(kernel_groups), kernel_channels_(kernel_channels), kernel_h_(kernel_h), 
		kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w), padding_h_(padding_h), 
		padding_w_(padding_w), lambda_(lambda), alpha_(alpha),
		use_biases_(use_biases), kernels_filler_(weights_filler), biases_filler_(biases_filler){}
	int kernel_groups_ = 1;
	int kernel_channels_ = 1;
	int kernel_h_ = 1;
	int kernel_w_ = 1;
	int stride_h_ = 1;
	int stride_w_ = 1;
	int padding_h_ = 0;
	int padding_w_ = 0;
	dtype lambda_ = 1.0;
	dtype alpha_ = 1.0;
	bool use_biases_ = true;
	FillerParameters kernels_filler_ = FillerParameters::NORMAL_DISTRIBUTION;
	FillerParameters biases_filler_ = FillerParameters::NORMAL_DISTRIBUTION;
	std::shared_ptr<Tensor<dtype>> kernels_ = nullptr;
	std::shared_ptr<Tensor<dtype>> biases_ = nullptr;
};

template<typename dtype>
struct InnerProductLayerParams {
	InnerProductLayerParams() = default;
	InnerProductLayerParams(const dtype lambda, const dtype alpha, bool use_biases,
		const FillerParameters& weights_filler, const FillerParameters& biases_filler) :
		lambda_(lambda), alpha_(alpha), use_biases_(use_biases), weights_filler_(weights_filler),
		biases_filler_(biases_filler){}
	dtype lambda_ = 1.0;
	dtype alpha_ = 1.0;
	bool use_biases_ = true;
	FillerParameters weights_filler_ = FillerParameters::NORMAL_DISTRIBUTION;
	FillerParameters biases_filler_ = FillerParameters::NORMAL_DISTRIBUTION;
	std::shared_ptr<Tensor<dtype>> weights_ = nullptr;
	std::shared_ptr<Tensor<dtype>> biases_ = nullptr;
};

template<typename dtype>
struct LayerParamsManage {
	std::string type_;
	bool use_gpu_;
	PoolingLayerParams<dtype> pooling_layer_params;
	ConvLayerParams<dtype> conv_layer_params_;
	InnerProductLayerParams<dtype> inner_product_layer_params_;
};

}




#endif
