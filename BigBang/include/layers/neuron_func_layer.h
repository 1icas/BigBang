#ifndef NEURON_FUNC_LAYER_H
#define NEURON_FUNC_LAYER_H

#include "layer_type_macro.h"
#include "../layer.h"

namespace BigBang {

template<typename dtype>
class NeuronFuncLayer : public Layer<dtype> {
public:
	NeuronFuncLayer(const LayerParameter& params) :
		Layer(params) {}

	virtual ~NeuronFuncLayer() {}

	virtual inline const char* FunctionType() const override {
		return NEURON_FUNC_TYPE;
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override = 0;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override = 0;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override = 0;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override = 0;
};

}


#endif
