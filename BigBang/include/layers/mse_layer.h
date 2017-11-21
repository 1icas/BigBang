#ifndef MSE_LAYER_H
#define MSE_LAYER_H

#include <memory>

#include "cost_func_layer.h"
#include "layer_type_macro.h"
#include "../tensor.h"

namespace BigBang {
template<typename dtype>
class MSELayer : public CostFuncLayer<dtype> {
public:
	MSELayer(const LayerParameter& params)
		: CostFuncLayer<dtype>(params) {
	}
	virtual ~MSELayer() {}

	virtual inline const char* Type() const override {
		return MSE_LAYER_TYPE;
	}
	virtual void SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) override;

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;

private:
	virtual void Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top);
};
}


#endif
