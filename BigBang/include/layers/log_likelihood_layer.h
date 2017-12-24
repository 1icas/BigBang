#ifndef LOG_LIKELIHOOD_LAYER_H
#define LOG_LIKELIHOOD_LAYER_H

#include <memory>
#include "cost_func_layer.h"
#include "layer_type_macro.h"
#include "../tensor.h"

namespace BigBang {
template<typename dtype>
class LogLikelihoodLayer : public CostFuncLayer<dtype> {
public:
	LogLikelihoodLayer(const LayerParameter& params)
		: CostFuncLayer<dtype>(params) {
	}
	virtual ~LogLikelihoodLayer() {}

	virtual inline const char* Type() const override {
		return LOG_LIKELIHOOD;
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	//TODO: i will implment this function soon
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};
	//TODO:
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override
	{
		Forward_CPU(bottom, top);
	}
	//TODO: i will implment this function soon
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};
	virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;

private:
	std::shared_ptr<Tensor<dtype>> result_;
};
}










#endif
