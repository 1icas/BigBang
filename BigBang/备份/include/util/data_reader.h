#ifndef DATA_READER_H
#define DATA_READER_H

#include <iostream>
#include <memory>

#include "../tensor.h"

//it's only support to read image data now

namespace BigBang {

struct DataParams {
	enum DataSet {
		Mnist
	};
	DataParams(const DataSet ds, const int channels, const int w, const int h, const std::string& train_data_image_dir,
		const std::string& train_data_label_dir, const int begin_train_data_index, const int end_train_data_index, bool use_validation_data,
		const std::string& validation_data_image_dir, const std::string& validation_data_label_dir, 
		const int begin_validation_data_index,
		const int end_validataion_data_index, bool use_test_data, 
		const std::string& test_data_image_dir,
		const std::string& test_data_label_dir,
		const int begin_test_data_index, const int end_test_data_index)
		: ds_(ds), channels_(channels), w_(w), h_(h), 
		train_data_image_dir_(train_data_image_dir), 
		train_data_label_dir_(train_data_label_dir), 
		begin_train_data_index_(begin_train_data_index), 
		end_train_data_index_(end_train_data_index), 
		use_validation_data_(use_validation_data),
		validation_data_image_dir_(validation_data_image_dir), 
		validation_data_label_dir_(validation_data_label_dir),
		begin_validation_data_index_(begin_validation_data_index),
		end_validataion_data_index_(end_validataion_data_index), 
		use_test_data_(use_test_data), 
		test_data_image_dir_(test_data_image_dir), 
		test_data_label_dir_(test_data_label_dir),
		begin_test_data_index_(begin_test_data_index), 
		end_test_data_index_(end_test_data_index){}
	DataSet ds_;
	int channels_;
	int w_;
	int h_;
	std::string train_data_image_dir_;
	std::string train_data_label_dir_;
	int begin_train_data_index_;
	int end_train_data_index_;
	bool use_validation_data_;
	std::string validation_data_image_dir_;
	std::string validation_data_label_dir_;
	int begin_validation_data_index_;
	int end_validataion_data_index_;
	bool use_test_data_;
	std::string test_data_image_dir_;
	std::string test_data_label_dir_;
	int begin_test_data_index_;
	int end_test_data_index_;
};

template<typename dtype>
class DataReader {
public:
	DataReader(const DataParams& params)
	: params_(params), train_data_image_(nullptr), train_data_label_(nullptr), validation_data_image_(nullptr),
		validation_data_label_(nullptr), test_data_image_(nullptr), test_data_label_(nullptr){
		Init();
	}

	void Read();

	const std::shared_ptr<Tensor<dtype>> GetTrainDataImage() const {
		return train_data_image_;
	}

	const std::shared_ptr<Tensor<dtype>> GetTrainDataLabel() const {
		return train_data_label_;
	}

	const std::shared_ptr<Tensor<dtype>> GetValidationDataImage() const {
		return validation_data_image_;
	}

	const std::shared_ptr<Tensor<dtype>> GetValidationDataLabel() const {
		return validation_data_label_;
	}

	const std::shared_ptr<Tensor<dtype>> GetTestDataImage() const {
		return test_data_image_;
	}

	const std::shared_ptr<Tensor<dtype>> GetTestDataLabel() const {
		return test_data_label_;
	}

private:
	void Init();

private:
	DataParams params_;

	std::shared_ptr<Tensor<dtype>> train_data_image_;
	std::shared_ptr<Tensor<dtype>> train_data_label_;
	std::shared_ptr<Tensor<dtype>> validation_data_image_;
	std::shared_ptr<Tensor<dtype>> validation_data_label_;
	std::shared_ptr<Tensor<dtype>> test_data_image_;
	std::shared_ptr<Tensor<dtype>> test_data_label_;
};
}



#endif
