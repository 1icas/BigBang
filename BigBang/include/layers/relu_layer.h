#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include "activation_func_layer.h"
#include "layer_type_macro.h"
#include "../tensor.h"

namespace BigBang {
	template<typename dtype>
	class ReluLayer : public ActivationFuncLayer<dtype> {
	public:
		ReluLayer(const LayerParameter& params) :ActivationFuncLayer(params) {}
		virtual ~ReluLayer() {}
		virtual inline const char* Type() const override { return RELU_LAYER_TYPE; }

	protected:
		virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
		virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
		virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	};
}




#endif
