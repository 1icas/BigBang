#ifndef IMAGE_LAYER_H
#define IMAGE_LAYER_H

#include "image_func_layer.h"
#include "layer_type_macro.h"

namespace BigBang {

template<typename dtype>
class ImageLayer : public ImageFuncLayer<dtype> {
public:
	ImageLayer(const LayerParameter& params)
		: ImageFuncLayer(params)/*, params_(params.image_layer_params_),
		batch_size_(params_.batch_size_)*/{}
	virtual ~ImageLayer() {}

	virtual inline const char* Type() const override {
		return IMAGE_TYPE;
	}

protected:
	virtual void Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
	virtual void Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) override;
private:
//	ImageLayerParams params_;
//	int batch_size_;
};

}














#endif
