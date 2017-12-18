#ifndef DATA_PREPROCESS_H
#define DATA_PREPROCESS_H

#include <memory>
#include <string>
#include "../tensor.h"
#include "../../proto/bigbang.pb.h"

namespace BigBang {

template<typename dtype>
class DataPreprocess {
public:
	DataPreprocess(const DataPreprocessParameter& params)
	: params_(params) {
		Init();
	}

	void Preprocess(const std::string& row_data, dtype* ripe_data);
	void Preprocess(const Datum& datum, dtype* ripe_data);

private:
	void Init();

private:
	DataPreprocessParameter params_;
	std::shared_ptr<Tensor<dtype>> mean_;

	bool use_mean_ = false;
};

}





#endif
