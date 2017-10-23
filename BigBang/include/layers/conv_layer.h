#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <memory>

#include "layer_type_macro.h"
#include "neuron_func_layer.h"
#include "../layer_params_manage.h"
#include "../../include/tensor.h"


namespace BigBang {

template<typename dtype>
class ConvLayer : public NeuronFuncLayer<dtype> {
public:
	ConvLayer(const LayerParamsManage<dtype>& params) :
		NeuronFuncLayer(params), unroll_bottom_(nullptr) {
		conv_params_ = params.conv_layer_params_;
		kernel_groups_ = conv_params_.kernel_groups_;
		kernel_channels_ = conv_params_.kernel_channels_;
		kernel_h_ = conv_params_.kernel_h_;
		kernel_w_ = conv_params_.kernel_w_;
		stride_h_ = conv_params_.stride_h_;
		stride_w_ = conv_params_.stride_w_;
		padding_h_ = conv_params_.padding_h_;
		padding_w_ = conv_params_.padding_w_;
		lambda_ = conv_params_.lambda_;
		alpha_ = conv_params_.alpha_;
		use_biases_ = conv_params_.use_biases_;
		kernels_ = conv_params_.kernels_;
		biases_ = conv_params_.biases_;
	}

	virtual inline const char* Type() const override {
		return CONV_LAYER_TYPE;
	}

	virtual void SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) override;

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override {};
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};

private:

	// check the bootom tensor and top tensor format
	// guarantee all the tensor have the right dimension info
	void Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top);
	void Prepare(const Tensor<dtype>* bottom, const Tensor<dtype>* top);
	void UpdateParams(const dtype* bottom_data, const dtype* delta);

private:
	ConvLayerParams<dtype> conv_params_;
	std::shared_ptr<Tensor<dtype>> unroll_bottom_;
	int kernel_groups_;
	int kernel_channels_;
	int kernel_h_;
	int kernel_w_;
	int stride_h_;
	int stride_w_;
	int padding_h_;
	int padding_w_;
	dtype lambda_;
	dtype alpha_;
	bool use_biases_;
	std::shared_ptr<Tensor<dtype>> kernels_;
	std::shared_ptr<Tensor<dtype>> biases_;
	//TODO: unuse 
	int dilation_h_ = 0;
	int dilation_w_ = 0;

	//the data infomation
	int nums_;
	int bottom_channels_;
	int bottom_row_;
	int bottom_column_;
	int top_channels_;
	int top_row_;
	int top_column_;
	int biases_groups_;
	int biases_channels_;
};

}




#endif
