#ifndef LAYER_H
#define LAYER_H

#include <memory>
#include <vector>
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
	virtual void SetUp(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		VALIDATE_POINTER(bottom);
		VALIDATE_POINTER(top);
		Prepare(bottom, top);
		Check(bottom, top);
	}

	void Reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		VALIDATE_POINTER(bottom);
		VALIDATE_POINTER(top);
		CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
		CHECK_EQ(top->dimension(), DATA_DIMENSION);
		reshape(bottom, top);
	}

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

	std::shared_ptr<Tensor<dtype>>& Labels() {
		return labels_;
	}

	std::vector<std::shared_ptr<Tensor<dtype>>>& get_learnable_params() {
		return learnable_params_;
	}

	LayerParameter::Phase Phase() {
		return params_.phase();
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) = 0;

	virtual void Check(const Tensor<dtype>* bottom, const Tensor<dtype>* top) {}
	virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {}
	virtual void reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) = 0;
protected:
	LayerParameter params_;
	std::shared_ptr<Tensor<dtype>> labels_;
	std::vector<std::shared_ptr<Tensor<dtype>>> learnable_params_;

};

}



#endif
