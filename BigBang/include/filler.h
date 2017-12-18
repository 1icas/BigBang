#ifndef FILLER_H
#define FILLER_H

#include <vector>
#include <random>
#include "base.h"
#include "tensor.h"
#include "../proto/bigbang.pb.h"
#include "util/math_function_ptr.h"

namespace BigBang {

//on the cpu hardward
template<typename dtype>
class Filler {
public:
	explicit Filler(const FillerParameter& params)
		:params_(params){}
	virtual void Fill(Tensor<dtype>* t) = 0;

protected:
	FillerParameter params_;
};

template<typename dtype>
class GaussianDistributionFiller : public Filler<dtype> {
public:
	explicit GaussianDistributionFiller(const FillerParameter& params)
		: Filler(params){}
	virtual void Fill(Tensor<dtype>* t) override {
		VALIDATE_POINTER(t);
		const int size = t->size();
		dtype* data = t->mutable_cpu_data();
		//TODO: the param 1 and 2 are unused now
		GaussianDistribution<dtype>(params_.mean(), params_.std(), size, data);
	}
};

template<typename dtype>
class XavierFiller : public Filler<dtype> {
public:
	explicit XavierFiller(const FillerParameter& params)
		: Filler(params){}
	virtual void Fill(Tensor<dtype>* t) override {
		VALIDATE_POINTER(t);
		CHECK_EQ(t->dimension(), DATA_DIMENSION);
		dtype fan_in = t->size() / t->shape(0);
		dtype scale = sqrt(static_cast<dtype>(3) / fan_in);
		bigbang_cpu_random_uniform(t->size(), -scale, scale, t->mutable_cpu_data());
	}
};

template<typename dtype>
std::shared_ptr<Filler<dtype>> CreateFiller(const FillerParameter& params) {
	switch (params.type()) {
	case FillerParameter::GAUSSIAN_DISTRIBUTION:
		return std::make_shared<GaussianDistributionFiller<dtype>>(params);
		break;
	case FillerParameter::XAVIER:
		return std::make_shared<XavierFiller<dtype>>(params);
		break;
	default:
		THROW_EXCEPTION;
	}
	return nullptr;
}


}
#endif
