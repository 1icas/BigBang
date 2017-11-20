#include "../../include/util/data_reader.h"

#include "../../include/base.h"
#include "../../include/util/load_data.h"

namespace BigBang {

template<typename dtype>
void DataReader<dtype>::Read() {
	switch(params_.ds_) {
		case DataParams::DataSet::Mnist:
			ReadMnistImage<dtype>(params_.train_data_image_dir_, params_.begin_train_data_index_, params_.end_train_data_index_,
				params_.channels_, params_.h_, params_.w_, train_data_image_.get());
			ReadMnistLabel<dtype>(params_.train_data_label_dir_, params_.begin_train_data_index_, params_.end_train_data_index_,
				params_.channels_, params_.h_, params_.w_, train_data_label_.get());
			if (params_.use_test_data_) {
				ReadMnistImage<dtype>(params_.test_data_image_dir_, params_.begin_test_data_index_, params_.end_test_data_index_,
					params_.channels_, params_.h_, params_.w_, test_data_image_.get());
				ReadMnistLabel<dtype>(params_.test_data_label_dir_, params_.begin_test_data_index_, params_.end_test_data_index_,
					params_.channels_, params_.h_, params_.w_, test_data_label_.get());
			}
			break;
		default:
			THROW_EXCEPTION;
	}
}

template<typename dtype>
void DataReader<dtype>::Init() {
	const int train_nums = params_.end_train_data_index_ - params_.begin_train_data_index_;
	const int channels = params_.channels_;
	const int h = params_.h_;
	const int w = params_.w_;
	train_data_image_ = std::make_shared<Tensor<dtype>>(
		std::vector<int>{train_nums, channels, h, w});
	train_data_label_ = std::make_shared<Tensor<dtype>>(
		std::vector<int>{train_nums, channels, 1, 10});
	if (params_.use_validation_data_) {
		const int validation_nums = params_.end_validataion_data_index_ - params_.begin_validation_data_index_;
		validation_data_image_ = std::make_shared<Tensor<dtype>>(
			std::vector<int>{validation_nums, channels, h, w});
		validation_data_label_ = std::make_shared<Tensor<dtype>>(
			std::vector<int>{validation_nums, channels, 1, 10});
	}
	if (params_.use_test_data_) {
		const int test_nums = params_.end_test_data_index_ - params_.begin_test_data_index_;
		test_data_image_ = std::make_shared<Tensor<dtype>>(
			std::vector<int>{test_nums, channels, h, w});
		test_data_label_ = std::make_shared<Tensor<dtype>>(
			std::vector<int>{test_nums, channels, 1, 10});
	}
	

}

INSTANTIATE_CLASS(DataReader);


}