#ifndef SOFTMAX_COST_LAYER_H
#define SOFTMAX_COST_LAYER_H

#include <memory>

#include "cost_func_layer.h"
#include "layer_type_macro.h"
#include "../tensor.h"

namespace BigBang {
	template<typename dtype>
	class SoftmaxCostLayer : public CostFuncLayer<dtype> {
	public:
		SoftmaxCostLayer(const LayerParameter& params)
			: CostFuncLayer<dtype>(params) {
		}
		virtual ~SoftmaxCostLayer() {}

		virtual inline const char* Type() const override {
			return SOFTMAX_COST_LAYER_TYPE;
		}

	protected:
		virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
		virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
		virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
		virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;

	private:
		std::shared_ptr<Tensor<dtype>> softmax_result_;
		std::shared_ptr<Tensor<dtype>> softmax_sum_;
		std::shared_ptr<Tensor<dtype>> max_num_;
	};
}








#endif
