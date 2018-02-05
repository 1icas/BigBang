#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "activation_func_layer.h"
#include "layer_type_macro.h"
#include "../tensor.h"

namespace BigBang {

template<typename dtype>
class SoftmaxLayer : public ActivationFuncLayer<dtype> {
public:
	SoftmaxLayer(const LayerParameter& params) :ActivationFuncLayer(params) {}
	virtual ~SoftmaxLayer() {}
	virtual inline const char* Type() const override { return SOFTMAX_LAYER_TYPE; }
protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	//i will implement this function in the furture
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};
	virtual	void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;

private:
	std::shared_ptr<Tensor<dtype>> softmax_sum_;
	std::shared_ptr<Tensor<dtype>> max_num_;
};
}







#endif
