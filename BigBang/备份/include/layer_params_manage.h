#ifndef LAYER_PARAMS_MANAGE_H
#define LAYER_PARAMS_MANAGE_H

#include <memory>
#include <string>

#include "filler.h"
#include "tensor.h"

namespace BigBang {

template<typename dtype>
struct PoolingLayerParams {
	enum class Pool {
		MaxPool,
		AveragePool
	};

	PoolingLayerParams() = default;
	PoolingLayerParams(const Pool& pool, const int pool_h, const int pool_w, const int stride_h,  const int stride_w) :
		pool_(pool), pool_h_(pool_h), pool_w_(pool_w), stride_h_(stride_h), stride_w_(stride_w) {}

	Pool pool_ = Pool::MaxPool;
	int pool_h_ = 1;
	int pool_w_ = 1;
	int stride_h_ = 1;
	int stride_w_ = 1;
};

template<typename dtype>
struct ConvLayerParams {
	ConvLayerParams() = default;
	ConvLayerParams(const int kernel_groups, const int kernel_channels, const int kernel_h, 
		const int kernel_w, const int stride_h, const int stride_w, const int padding_h, const int padding_w, const dtype lambda, 
		const dtype alpha, bool use_biases, const FillerParams<dtype>& weights_filler,
		const FillerParams<dtype>& biases_filler, const std::shared_ptr<Tensor<dtype>> kernels,
		const std::shared_ptr<Tensor<dtype>> biases) : kernel_groups_(kernel_groups), kernel_channels_(kernel_channels), kernel_h_(kernel_h),
		kernel_w_(kernel_w), stride_h_(stride_h), stride_w_(stride_w), padding_h_(padding_h), 
		padding_w_(padding_w), lambda_(lambda), alpha_(alpha),
		use_biases_(use_biases), kernels_filler_(weights_filler), biases_filler_(biases_filler),
		kernels_(kernels), biases_(biases) {
		filler_params();
	}
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
	FillerParams<dtype> kernels_filler_;
	FillerParams<dtype> biases_filler_;
	std::shared_ptr<Tensor<dtype>> kernels_ = nullptr;
	std::shared_ptr<Tensor<dtype>> biases_ = nullptr;

	void filler_params() {
		if (kernels_filler_.type_ != FillerParams<dtype>::FillerType::UNUSED) {
			CreateFiller<dtype>(kernels_filler_)->Fill(kernels_.get());
		}
		if (use_biases_ &&
			kernels_filler_.type_ != FillerParams<dtype>::FillerType::UNUSED) {
			CreateFiller<dtype>(biases_filler_)->Fill(kernels_.get());
		}
	}
};

template<typename dtype>
struct InnerProductLayerParams {
	InnerProductLayerParams() = default;
	InnerProductLayerParams(const dtype lambda, const dtype alpha, bool use_biases,
		const FillerParams<dtype>& weights_filler, const FillerParams<dtype>& biases_filler,
		const std::shared_ptr<Tensor<dtype>>& weights, const std::shared_ptr<Tensor<dtype>>& biases) :
		lambda_(lambda), alpha_(alpha), use_biases_(use_biases), weights_filler_(weights_filler),
		biases_filler_(biases_filler), weights_(weights), biases_(biases){
		filler_params();
	}
	dtype lambda_ = 1.0;
	dtype alpha_ = 1.0;
	bool use_biases_ = true;
	FillerParams<dtype> weights_filler_;
	FillerParams<dtype> biases_filler_;
	std::shared_ptr<Tensor<dtype>> weights_ = nullptr;
	std::shared_ptr<Tensor<dtype>> biases_ = nullptr;

	void filler_params() {
		if (weights_filler_.type_ != FillerParams<dtype>::FillerType::UNUSED) {
			CreateFiller<dtype>(weights_filler_)->Fill(weights_.get());
		}
		if (use_biases_
			&& biases_filler_.type_ != FillerParams<dtype>::FillerType::UNUSED) {
			CreateFiller<dtype>(biases_filler_)->Fill(biases_.get());
		}
	}
};

template<typename dtype>
struct CostFuncLayerParams {
	CostFuncLayerParams() = default;
};

template<typename dtype>
struct MnistImageLayerParams {
	MnistImageLayerParams() = default;
	MnistImageLayerParams(std::string train_data_dir, std::string validation_data_dir,
		std::string test_data_dir, int train_data_nums, int validation_data_nums,
		int test_data_nums, int channels, int w, int h) :
		train_data_dir_(train_data_dir), validation_data_dir_(validation_data_dir),
		test_data_dir_(test_data_dir), train_data_nums_(train_data_nums),
		validation_data_nums_(validation_data_nums), test_data_nums_(test_data_nums),
		channels_(channels), w_(w), h_(h) {}
	std::string train_data_dir_;
	std::string validation_data_dir_;
	std::string test_data_dir_;
	int train_data_nums_;
	int validation_data_nums_;
	int test_data_nums_;
	int channels_;
	int w_;
	int h_;
};

template<typename dtype>
struct ImageLayerParams {
	ImageLayerParams() = default;
	ImageLayerParams(const int batch_size) :
		batch_size_(batch_size) {}

	int batch_size_;
};

template<typename dtype>
struct LayerParamsManage {
	std::string type_;
	bool use_gpu_;
	PoolingLayerParams<dtype> pooling_layer_params_;
	ConvLayerParams<dtype> conv_layer_params_;
	InnerProductLayerParams<dtype> inner_product_layer_params_;
	CostFuncLayerParams<dtype> cost_func_layer_params_;
	MnistImageLayerParams<dtype> mnist_image_layer_params_;
	ImageLayerParams<dtype> image_layer_params_;
};

}




#endif
