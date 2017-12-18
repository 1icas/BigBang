#include "../include/solver.h"

#include <iostream>

#include "../include/config.h"
#include "../include/gtest.h"
#include "../include/util/math_function_ptr.h"

namespace BigBang {
	
template<typename dtype>
void Solver<dtype>::Run() {
	
}

template<typename dtype>
void Solver<dtype>::Train() {
	auto net = nets_[NetParameter::TRAIN];
	if (!net) THROW_EXCEPTION;
	const int train_iters = params_.train_iterations();
	for (int i = 0; i < train_iters; ++i) {
		net->Train();
		auto times = params_.test_validatedata_accuracy_per_train_iterations();
		if (i != 0 && times != 0 && i % times == 0) {
			Validate();
		}

		UpdateLearnableParams();
	}

}

template<typename dtype>
void Solver<dtype>::Validate() {
	auto net = nets_[NetParameter::VALIDATE];
	if (!net) THROW_EXCEPTION;
	const int validate_iters = params_.validate_iterations();
	const int batch_size = params_.validate_batch_size();
	int count = 0;
	for (int i = 0; i < validate_iters; ++i) {
		count += net->TestValidateData();
	}

	const dtype percent = static_cast<dtype>(count) / static_cast<dtype>(batch_size*validate_iters);
	std::cout << "the validate data accuracy is : " << percent << std::endl;
}

template<typename dtype>
void Solver<dtype>::UpdateLearnableParams() {
	bool is_cpu = Config::Get().mode() == Config::ProcessUnit::CPU;
	const int size = learnable_params_.size();
	const dtype learn_rate = params_.lr();
	const int batch_size = params_.train_batch_size();
	for (int i = 0; i < size; ++i) {
		for (int k = 0; k < learnable_params_[i].size(); ++k) {
			auto lp = learnable_params_[i][k];
			//the zero index is weight
			if (k == 0) {
				WeightDecay(lp);
			}
			if (is_cpu) {
				const dtype* diff_data = lp->cpu_diff_data();
				dtype* data = lp->mutable_cpu_data();
				bigbang_cpu_minus<dtype>(data, diff_data, lp->size(), learn_rate / batch_size, data);
				/*for (int i = 0; i < lp->size(); ++i) {
					std::cout << diff_data[i] << std::endl;
				}*/
			}
			else {
				const dtype* diff_data = lp->gpu_diff_data();
				dtype* data = lp->mutable_gpu_data();
				bigbang_gpu_minus<dtype>(data, diff_data, lp->size(), learn_rate / batch_size, data);
			}
		}
	}
}

template<typename dtype>
void Solver<dtype>::WeightDecay(const std::shared_ptr<Tensor<dtype>>& weights) {
	auto wdp = params_.weight_decay_param();
	bool is_cpu = Config::Get().mode() == Config::ProcessUnit::CPU;
	const dtype alpha = static_cast<dtype>(wdp.alpha());
	const int size = weights->size();
	dtype* weights_data = is_cpu ? weights->mutable_cpu_data() : weights->mutable_gpu_data();
	switch (wdp.method()) {
	case WeightDecayParameter::L2REGULARIZATION:
		if (is_cpu) {
			bigbang_cpu_minus<dtype>(weights_data, weights_data, size, alpha, weights_data);
		}
		else {
			bigbang_gpu_minus<dtype>(weights_data, weights_data, size, alpha, weights_data);
		}
		break;
	default: break;
	}
}


INSTANTIATE_CLASS(Solver);
}