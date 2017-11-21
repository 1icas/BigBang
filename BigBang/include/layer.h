#ifndef LAYER_H
#define LAYER_H

#include "config.h"
#include "tensor.h"
#include "../proto/bigbang.pb.h"

namespace BigBang {

template<typename dtype>
class Layer {
public:
	Layer(const LayerParameter& params)
		:params_(params) {

	}
	
	virtual ~Layer(){}

	virtual inline const char* FunctionType() const { return " "; }
	virtual inline const char* Type() const { return " "; }
	virtual void SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) = 0;
	void Forward(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		if (Config::Get().mode() == Config::ProcessUnit::GPU) {
			Forward_GPU(bottom, top);
		}
		else {
			Forward_CPU(bottom, top);
		}
	}
	void Backward(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
		if (Config::Get().mode() == Config::ProcessUnit::GPU) {
			Backward_GPU(top, bottom);
		}
		else {
			Backward_CPU(top, bottom);
		}
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;

protected:
	LayerParameter params_;
};

}



#endif
