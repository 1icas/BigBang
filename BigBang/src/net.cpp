#include "../include/net.h"

#include "../include/base.h"
#include "../include/layer_factory.h"
#include "../include/layers/layer_type_macro.h"

namespace BigBang {

template<typename dtype>
void Net<dtype>::GenerateLayers() {
	for (auto& it : manages_) {
		layers_.push_back(LayerRegistry<dtype>::CreateLayer(it));
	}
}

template<typename dtype>
void Net<dtype>::GenerateInput() {
	for (auto& v : layers_) {
		if (v->FunctionType() == IMAGE_FUNC_TYPE || v->FunctionType() == NEURON_FUNC_TYPE) {
			input_.push_back(std::make_shared<Tensor<dtype>>());
		}
	}
}

template<typename dtype>
void Net<dtype>::TrainDebug() {

}

INSTANTIATE_CLASS(Net);
}