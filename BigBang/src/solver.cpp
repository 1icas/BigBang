#include "../include/solver.h"

#include <iostream>

#include "../include/config.h"
#include "../include/gtest.h"
#include "../include/util/math_function_ptr.h"
#include "../include/util/parse.h"

namespace BigBang {
	
template<typename dtype>
Solver<dtype>::Solver(const SolverParameter& params) 
	: params_(params) {
	if (params.mode() == SolverParameter::CPU) {
		Config::Get().set_mode(Config::ProcessUnit::CPU);
	}
	else {
		Config::Get().set_mode(Config::ProcessUnit::GPU);
	}
	const int net_size = params.net_param_size();
	for (int i = 0; i < net_size; ++i) {
		const int state = params.net_param(i).state();
		nets_.insert(std::make_pair(state,
			std::make_shared<Net<dtype>>(params.net_param(i))));
		auto layers_ = nets_[state]->Layers();
		const int size = layers_.size();
		if (params.net_param(i).state() == NetParameter::TRAIN) {
			for (int k = 0; k < size; ++k) {
				learnable_params_.push_back(layers_[k]->get_learnable_params());
			}
		}
		else {
			//the other net should be shared the train net learnable parameter
			CHECK_NE(learnable_params_.size(), 0);
			for (int k = 0; k < size; ++k) {
				if (layers_[k]->get_learnable_params().size() != 0) {
					layers_[k]->get_learnable_params() = learnable_params_[k];
				}
			}
		}
	}
	MomentumParamsInit();
	DeserializeNumerical();
}

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

		MomentumProcess();
		UpdateLearnableParams();
	}
	SerializeNumerical();
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
	dtype learn_rate = params_.lr();
	const int batch_size = params_.train_batch_size();
	for (int i = 0; i < size; ++i) {
		for (int k = 0; k < learnable_params_[i].size(); ++k) {
			auto lp = learnable_params_[i][k];
			auto mp = momentum_params_[i][k];
			//the zero index is weight
			if (k == 0) {
				WeightDecay(lp);
			}
			if (is_cpu) {
				//TODO: mp->cpu_diff_data()
				const dtype* diff_data = params_.momentum_ratio() == 0. ?
					lp->cpu_diff_data() : mp->cpu_data();
				dtype* data = lp->mutable_cpu_data();
				bigbang_cpu_minus<dtype>(data, diff_data, lp->size(), learn_rate / batch_size, data);
				/*for (int i = 0; i < lp->size(); ++i) {
					std::cout << diff_data[i] << std::endl;
				}*/
			}
			else {
				const dtype* diff_data = params_.momentum_ratio() == 0. ? 
					lp->gpu_diff_data() : mp->gpu_data();
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

template<typename dtype>
void Solver<dtype>::MomentumParamsInit() {
	int n = learnable_params_.size();
	for (int i = 0; i < n; ++i) {
		std::vector<std::shared_ptr<Tensor<dtype>>> param;
		for (int k = 0; k < learnable_params_[i].size(); ++k) {
			param.push_back(std::make_shared<Tensor<dtype>>(learnable_params_[i][k]->shape()));
		}
		momentum_params_.push_back(param);
	}
}

template<typename dtype>
void Solver<dtype>::MomentumProcess() {
	dtype ratio = static_cast<dtype>(params_.momentum_ratio());
	if (ratio <= 0) return;
	bool is_cpu = Config::Get().mode() == Config::ProcessUnit::CPU;
	for (int i = 0; i < momentum_params_.size(); ++i) {
		auto params = momentum_params_[i];
		for (int k = 0; k < params.size(); ++k) {
			if (is_cpu) {
				bigbang_cpu_plus(learnable_params_[i][k]->cpu_diff_data(), params[k]->size(), static_cast<dtype>(1.) - ratio,
					ratio, params[k]->mutable_cpu_data());
			}
			else {
				bigbang_gpu_plus(learnable_params_[i][k]->gpu_diff_data(), params[k]->size(), static_cast<dtype>(1.) - ratio,
					ratio, params[k]->mutable_gpu_data());
			}
		}
	}
}

template<typename dtype>
void Solver<dtype>::SerializeNumerical() {
	const std::string write_model_dir = params_.write_model_dir();
	if (write_model_dir.empty()) return;
	TensorProtoVector tpv;
	auto learn_param_serialize = [&](const std::vector<std::vector<std::shared_ptr<Tensor<dtype>>>>& p) {
		for (int i = 0; i < p.size(); ++i) {
			for (int k = 0; k < p[i].size(); ++k) {
				TensorProto* tp = tpv.add_tensor();
				p[i][k]->Serialize(tp);
			}
		}
	};
	//serialize the learnable_params
	learn_param_serialize(learnable_params_);
	//serialize the momentum_params
	dtype ratio = static_cast<dtype>(params_.momentum_ratio());
	if (ratio > 0) {
		learn_param_serialize(momentum_params_);
	}
	//write the tensorprotovector in the file
	ParseMessageToBinaryFile(write_model_dir, tpv);
}

template<typename dtype>
void Solver<dtype>::DeserializeNumerical() {
	const std::string read_model_dir = params_.read_model_dir();
	if (read_model_dir.empty()) return;
	TensorProtoVector tpv;
	ParseBinaryFileToMessage(read_model_dir, &tpv);
	int index = 0;
	auto learn_param_deserialize = [&](std::vector<std::vector<std::shared_ptr<Tensor<dtype>>>>& p) {
		for (int i = 0; i < p.size(); ++i) {
			for (int k = 0; k < p[i].size(); ++k) {
				p[i][k]->Deserialize(&(tpv.tensor(index++)));
			}
		}
	};
	//deserialize the learnable_params
	learn_param_deserialize(learnable_params_);
	//deserialize the momentum_params
	dtype ratio = static_cast<dtype>(params_.momentum_ratio());
	if (ratio > 0) {
		learn_param_deserialize(momentum_params_);
	}
}


INSTANTIATE_CLASS(Solver);
}