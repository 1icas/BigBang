#ifndef POOLING_LAYER_H
#define POOLING_LAYER_H


#include <memory>

#include "layer_type_macro.h"
#include "neuron_func_layer.h"
#include "../layer.h"
#include "../../proto/bigbang.pb.h"


namespace BigBang {

// TODO:only support maxpool now
template<typename dtype>
class PoolingLayer : public NeuronFuncLayer<dtype> {
public:
	PoolingLayer(const LayerParameter& params) :
		NeuronFuncLayer(params), max_pool_pos_() {
		pooling_layer_params_ = params.pooling_layer_param();
		pooling_method_ = pooling_layer_params_.pool();
		//pool_ = pooling_layer_params_.pool_;
		pool_h_ = pooling_layer_params_.kernel_h();
		pool_w_ = pooling_layer_params_.kernel_w();
		stride_h_ = pooling_layer_params_.stride_h();
		stride_w_ = pooling_layer_params_.stride_w();
	}
	virtual ~PoolingLayer() {}

	virtual inline const char* Type() const override {
		return POOLING_LAYER_TYPE;
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) override;
	virtual void Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	void do_reshape(const Tensor<dtype>* bottom, Tensor<dtype>* top);
private:
	PoolingLayerParameter_PoolingMethod pooling_method_;
	PoolingLayerParameter pooling_layer_params_;
	std::shared_ptr<Tensor<int>> max_pool_pos_;
	//typename PoolingLayerParams<dtype>::Pool pool_;
	int pool_h_;
	int pool_w_;
	int stride_h_;
	int stride_w_;
	//the bottom and top data information
	int nums_;
	int bottom_channels_;
	int bottom_row_;
	int bottom_column_;
	int top_channels_;
	int top_row_;
	int top_column_;
};


}


#endif