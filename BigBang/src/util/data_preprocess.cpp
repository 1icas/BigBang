#include "../../include/util/data_preprocess.h"
#include <vector>
#include "../../include/base.h"
#include "../../include/util/parse.h"

namespace BigBang {

template<typename dtype>
void DataPreprocess<dtype>::Init() {
	const std::string& mean_file_path = params_.mean_file();
	if (!mean_file_path.empty()) {
		TensorProto tp;
		ParseBinaryFileToMessage(mean_file_path, &tp);
		auto shape = tp.shape();
		const int n = shape.dim_size();
		CHECK_NE(n, 0);
		int size = 1;
		for (int i = 0; i < n; ++i) {
			size *= shape.dim(i);
		}
		mean_.reset(new Tensor<dtype>(std::vector<int>{size}));
		dtype* mutable_mean_data = mean_->mutable_cpu_data();
		for (int i = 0; i < size; ++i) {
			mutable_mean_data[i] = static_cast<dtype>(tp.f_data(i));
		}
		use_mean_ = true;
	}
}

template<typename dtype>
void DataPreprocess<dtype>::Preprocess(const std::string& row_data, dtype* ripe_data) {
	const int size = row_data.size();
	const dtype scale = static_cast<dtype>(params_.scale());
	dtype* mean_data;
	if (use_mean_) {
		mean_data = mean_->mutable_cpu_data();
	}
	for (int k = 0; k < size; ++k) {
		dtype v = static_cast<dtype>(static_cast<unsigned char>(row_data[k]));
		if (use_mean_) {
			ripe_data[k] = (v - mean_data[k]) * scale;
		}
		else {
			ripe_data[k] = v * scale;
		}
	}
}

template<typename dtype>
void DataPreprocess<dtype>::Preprocess(const Datum& datum, dtype* ripe_data) {
	const int size = datum.f_data_size();
	const dtype scale = static_cast<dtype>(params_.scale());
	dtype* mean_data;
	if (use_mean_) {
		mean_data = mean_->mutable_cpu_data();
	}
	for (int k = 0; k < size; ++k) {
		dtype v = static_cast<dtype>(datum.f_data(k));
		if (use_mean_) {
			ripe_data[k] = (v - mean_data[k]) * scale;
		}
		else {
			ripe_data[k] = v * scale;
		}
	}
}

INSTANTIATE_CLASS(DataPreprocess);
}