#ifndef LAYER_H
#define LAYER_H

#include "layer_params_manage.h"

namespace BigBang {

template<typename dtype>
class Layer {
public:
	Layer(const LayerParamsManage<dtype>& params)
		:params_(params) {
		use_gpu_ = params.use_gpu_;
		if (use_gpu_) {
#ifdef CPU_ONLY
			NO_GPU;
#endif
		}
	}
	
	virtual ~Layer(){}

	virtual inline const char* FunctionType() const { return " "; }
	virtual inline const char* Type() const { return " "; }
	virtual void SetUp(const Tensor<dtype>* bottom, const Tensor<dtype>* top) = 0;
	void Forward(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		if (use_gpu_) {
			Forward_GPU(bottom, top);
		}
		else {
			Forward_CPU(bottom, top);
		}
	}
	void Backward(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
		if (use_gpu_) {
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
	LayerParamsManage<dtype> params_;
	bool use_gpu_;
};

}



#endif
