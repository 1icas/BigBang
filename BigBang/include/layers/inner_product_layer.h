#ifndef INNER_PRODUCT_LAYER_H
#define INNER_PRODUCT_LAYER_H

#include <memory>
#ifdef BIGBANG_TEST
#include <vector>
#endif

#include "neuron_func_layer.h"
#include "../layer.h"
#include "../tensor.h"
#include "../../proto/bigbang.pb.h"

#include <iostream>

namespace BigBang {

template<typename dtype>
class InnerProductLayer :public NeuronFuncLayer<dtype> {
public:
	InnerProductLayer(const LayerParameter& params)
		:NeuronFuncLayer(params) {
		inner_params_ = params.inner_product_layer_param();
		use_bias_ = inner_params_.use_bias();
	}
	virtual ~InnerProductLayer() {}

	virtual inline const char* Type() const override {
		return INNER_PRODUCT_LAYER_TYPE;
	}


#ifdef BIGBANG_TEST
	void printWeights() {
		for (int i = 0; i < weights_->size(); ++i) {
			std::cout << weights_->cpu_data()[i] << std::endl;
		}
	}

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
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
	virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) override;
	virtual void reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;

protected:
	std::shared_ptr<Tensor<dtype>> weights_;
	std::shared_ptr<Tensor<dtype>> biases_;

private:
	void UpdateParams_CPU(const dtype* bottom_data, const dtype* delta);
	void UpdateParams_GPU(const dtype* bottom_data, const dtype* delta);

private:
	InnerProductLayerParameter inner_params_;
	bool use_bias_;
	/*FillerParameter weight_filler_;
	FillerParameter bias_filler_;*/
	std::shared_ptr<Tensor<dtype>> middle_;

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

	//temp
	dtype alpha_ = 1.;

};

}



#endif
