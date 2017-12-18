#include "../../include/layers/data_layer.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "../../include/config.h"
#include "../../include/gtest.h"
#include "../../include/layer_factory.h"

namespace BigBang {

template<typename dtype>
void DataLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const std::string source = data_layer_params_.source();
	lmdb_.reset(new LMDB());
	lmdb_->Open(source, DBMode::READ);
	std::shared_ptr<Cursor> cursor(lmdb_->CreateCursor());
	CHECK_EQ(cursor->valid(), true);
	std::string value = cursor->value();
	Datum datum;
	datum.ParseFromString(value);
	const int channels = datum.channels();
	const int width = datum.width();
	const int height = datum.height();
	const int batch_size = data_layer_params_.batch_size();
	const int cache_count = data_layer_params_.cache_batch_count();
	top->Reshape(std::vector<int>{batch_size, channels, height, width});
	labels_.reset(new Tensor<dtype>(std::vector<int>{batch_size, 1, 1, 1}));
	for (int i = 0; i < cache_count; ++i) {
		ImageBlob<dtype> image_blob;
		image_blob.data_.reset(new Tensor<dtype>(std::vector<int>{batch_size, channels, height, width}));
		image_blob.labels_.reset(new Tensor<dtype>(std::vector<int>{batch_size, 1, 1, 1}));
		free_queue_.push(image_blob);
	}
	Start();
}

template<typename dtype>
void DataLayer<dtype>::entry() {
	bool use_gpu = Config::Get().mode() == Config::ProcessUnit::GPU;
	std::shared_ptr<Cursor> cursor(lmdb_->CreateCursor());
	const int batch_size = data_layer_params_.batch_size();
	cudaStream_t stream;
	if (use_gpu) {
		cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
	}

	while (!Should_Stop()) {
		ImageBlob<dtype> image_blob;
		free_queue_.wait_and_pop(image_blob);
		dtype* tensor_data = image_blob.data_->mutable_cpu_data();
		//TODO: also it's a optional label data, so what's the right time to use it?
		dtype* labels_data = image_blob.labels_->mutable_cpu_data();
		for (int i = 0; i < batch_size; ++i) {
			if (!cursor->valid()) cursor->SeekToFirst();
			std::string value = cursor->value();
			Datum datum;
			datum.ParseFromString(value);
			labels_data[i] = static_cast<dtype>(datum.label()); 
			const std::string& data = datum.data();
			if (!data.empty()) {
				const int n = data.size();
				data_preprocess_->Preprocess(data, tensor_data+i*n);
			}
			else {
				const int n = datum.f_data_size();
				data_preprocess_->Preprocess(datum, tensor_data+i*n);
			}
			cursor->Next();
		}
		if (use_gpu) {
			image_blob.data_->data()->async_gpu_data_push(stream);
			image_blob.labels_->data()->async_gpu_data_push(stream);
			cudaStreamSynchronize(stream);
		}
		full_queue_.push(image_blob);
	}

	if (use_gpu) {
		cudaStreamDestroy(stream);
	}
}

template<typename dtype>
void DataLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	if(current_image_blob_.data_)
		free_queue_.push(current_image_blob_);

	full_queue_.wait_and_pop(current_image_blob_);
	top->shared_data(*(current_image_blob_.data_.get()));
	labels_->shared_data(*(current_image_blob_.labels_.get()));
}

template<typename dtype>
void DataLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	Forward_CPU(bottom, top);
}

INSTANTIATE_CLASS(DataLayer);
REGISTRY_LAYER(Data);

}