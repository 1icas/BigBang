#ifndef FILLER_H
#define FILLER_H

#include <vector>

#include "base.h"
#include "tensor.h"
#include "../proto/bigbang.pb.h"
#include "util/math_function_ptr.h"

namespace BigBang {


template<typename dtype>
struct FillerParams {
	enum FillerType {
		UNUSED,
		GAUSSIAN_DISTRIBUTION
	};
	FillerParams() = default;
	FillerParams(const FillerType& type, const dtype mean, const dtype std)
		:type_(type), mean_(mean), std_(std){}

	FillerType type_ = FillerType::UNUSED;
	dtype mean_ = 0;
	dtype std_ = 1;
};


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
		GaussianDistribution<dtype>(0.0, 0.0, size, data);
	}
};

template<typename dtype>
std::shared_ptr<Filler<dtype>> CreateFiller(const FillerParameter& params) {
	switch (params.type()) {
	case FillerParameter::GAUSSIAN_DISTRIBUTION:
		return std::make_shared<GaussianDistributionFiller<dtype>>(params);
		break;
	default:
		THROW_EXCEPTION;
	}
	return nullptr;
}


}
#endif
