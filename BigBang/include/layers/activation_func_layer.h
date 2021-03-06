#ifndef ACTIVATION_FUNC_LAYER_H
#define ACTIVATION_FUNC_LAYER_H

#include "layer_type_macro.h"
#include "../layer.h"

namespace BigBang {
template<typename dtype>
class ActivationFuncLayer : public Layer<dtype>{
public:
	ActivationFuncLayer(const LayerParameter& params) :
		Layer(params) {}
	virtual ~ActivationFuncLayer() {}


	virtual inline const char* FunctionType() const override {
		return ACTIVATION_FUNC_TYPE;
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override  = 0;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override  = 0;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override = 0;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override  = 0;
};



}




#endif
