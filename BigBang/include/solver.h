#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <map>

#include "net.h"
#include "config.h"
#include "../proto/bigbang.pb.h"

namespace BigBang {
//sgd
template<typename dtype>
class Solver {
public:
	Solver(const SolverParameter& params)
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
	}

	void Run();

	void Train();
	void Validate();

private:
	void UpdateLearnableParams();
	void WeightDecay(const std::shared_ptr<Tensor<dtype>>& weights);

private:
	SolverParameter params_;
	std::map<int, std::shared_ptr<Net<dtype>>> nets_;
	std::vector<std::vector<std::shared_ptr<Tensor<dtype>>>> learnable_params_;
	//std::vector<std::shared_ptr<Net<dtype>>> nets_;
	//std::vector<std::shared_ptr<Tensor<dtype>>> 
};

}




#endif
