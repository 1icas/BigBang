#ifndef DATA_LAYER_H
#define DATA_LAYER_H

#include <memory>
#include "layer_type_macro.h"
#include "../layer.h"
#include "../thread.h"
#include "../thread_safe_queue.h"
#include "../../proto/bigbang.pb.h"
#include "../../include/util/data_preprocess.h"
#include "../../include/util/db_lmdb.h"

namespace BigBang {
template<typename dtype>
struct ImageBlob {
	std::shared_ptr<Tensor<dtype>> data_;
	std::shared_ptr<Tensor<dtype>> labels_;
};

template<typename dtype>
class DataLayer : public Layer<dtype>, Thread {
public:
	DataLayer(const LayerParameter& params)
		: Layer(params) {
		data_layer_params_ = params.data_layer_param();
		data_preprocess_.reset(new DataPreprocess<dtype>(data_layer_params_.preprocess_params()));
	}

	virtual ~DataLayer() {}

	virtual inline const char* Type() const { return DATA_LAYER; }

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override {};
	virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void entry();

private:
	bool Skip();

private:
	ThreadSafeQueue<ImageBlob<dtype>> free_queue_;
	ThreadSafeQueue<ImageBlob<dtype>> full_queue_;
	DataLayerParameter data_layer_params_;
	std::shared_ptr<DB> lmdb_;
	ImageBlob<dtype> current_image_blob_;
	std::shared_ptr<DataPreprocess<dtype>> data_preprocess_;
};

}







#endif
