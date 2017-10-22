#ifndef COST_FUNC_LAYER_H
#define COST_FUNC_LAYER_H

#include "layer_type_macro.h"
#include "../layer.h"

namespace BigBang {

template<typename dtype>
class CostFuncLayer : public Layer<dtype> {
public:
	CostFuncLayer(const LayerParamsManage<dtype>& params) :
		Layer(params) {}

	virtual inline const char* FunctionType() const override {
		return COST_FUNC_TYPE;
	}

	virtual void SetUp(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Forward(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;
};

}


#endif
