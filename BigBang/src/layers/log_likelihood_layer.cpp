#include "../../include/layers/log_likelihood_layer.h"
#include "../../include/layer_factory.h"


namespace BigBang {

template<typename dtype>
void LogLikelihoodLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	top->Reshape(std::vector<int>{bottom->shape(0), 1, 1, 1});
}

template<typename dtype>
void LogLikelihoodLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	//TODO: i will modify this soon.(the number 1)
	if (++count_ % 1 == 0) {
		const int nums = bottom->shape(0);
		const int per_data_size = bottom->size() / nums;
		const dtype* predict_result_data = bottom->cpu_data();
		const dtype* labels = top->cpu_data();
		dtype loss = 0;
		for (int i = 0; i < nums; ++i) {
			loss += -log(std::max<dtype>(predict_result_data[static_cast<int>(labels[i] + 0.1) + i*per_data_size], FLT_MIN));
		}
		std::cout << "loglikelihood training " << count_ << " times, the error is: " << loss / nums << std::endl;
	}
	
}

INSTANTIATE_CLASS(LogLikelihoodLayer);
REGISTRY_LAYER(LogLikelihood);
}