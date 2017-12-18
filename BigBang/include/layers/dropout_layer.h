#ifndef DROPOUT_LAYER_H
#define DROPOUT_LAYER_H

#include <memory>

#include "activation_func_layer.h"
#include "layer_type_macro.h"
#include "../tensor.h"

namespace BigBang {
	template<typename dtype>
	class DropoutLayer : public ActivationFuncLayer<dtype> {
	public:
		DropoutLayer(const LayerParameter& params) :ActivationFuncLayer(params) {
			dropout_params_ = params.dropout_layer_param();
		}
		virtual ~DropoutLayer() {}
		virtual inline const char* Type() const override { return DROPOUT_LAYER_TYPE; }

	protected:
		virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
		virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
		virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;

	private:
		DropoutLayerParameter dropout_params_;
		std::shared_ptr<Tensor<unsigned int>> mask_;
		dtype scale_;
		unsigned int threshold_;
	};
}












#endif
