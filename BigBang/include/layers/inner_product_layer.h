#ifndef INNER_PRODUCT_LAYER_H
#define INNER_PRODUCT_LAYER_H

#include <memory>
#ifdef _DEBUG
#include <vector>
#endif

#include "neuron_func_layer.h"
#include "../layer.h"
#include "../layer_params_manage.h"
#include "../tensor.h"

namespace BigBang {

template<typename dtype>
class InnerProductLayer :public NeuronFuncLayer<dtype> {
public:
	InnerProductLayer(const LayerParamsManage<dtype>& params)
		:NeuronFuncLayer(params) {
		params_ = params.inner_product_layer_params_;
		alpha_ = params_.alpha_;
		lambda_ = params_.lambda_;
		weights_ = params_.weights_;
		biases_ = params_.biases_;
	}

	virtual void SetUp(const Tensor<dtype>* bottom, Tensor<dtype>* top);

	virtual inline const char* Type() const override {
		return INNER_PRODUCT_LAYER_TYPE;
	}

#ifdef TEST
	std::vector<std::shared_ptr<Tensor<dtype>>> GetParams() {
		std::vector<std::shared_ptr<Tensor<dtype>>> test;
		test.push_back(weights_);
		test.push_back(biases_);
		return test;
	}
#endif

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override {};
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};

private:
	void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top);
	void Check(const Tensor<dtype>* bottom, Tensor<dtype>* top);
	void UpdateParams(const dtype* bottom_data, const dtype* delta);

private:
	InnerProductLayerParams<dtype> params_;

	bool use_biases_;
	dtype alpha_;
	dtype lambda_;
	std::shared_ptr<Tensor<dtype>> weights_;
	std::shared_ptr<Tensor<dtype>> biases_;
	//the data infomation
	int nums_;
	int bottom_row_;
	int bottom_column_;
	int top_row_;
	int top_column_;
	int weights_row_;
	int weights_column_;
	int biases_row_;
	int biases_column_;
};

}



#endif
