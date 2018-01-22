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
	Solver(const SolverParameter& params);

	void Run();
	void Train();
	void Validate();
	void SerializeNumerical();
	void DeserializeNumerical();

private:
	void UpdateLearnableParams();
	void WeightDecay(const std::shared_ptr<Tensor<dtype>>& weights);
	void MomentumParamsInit();
	void MomentumProcess();


private:
	SolverParameter params_;
	std::map<int, std::shared_ptr<Net<dtype>>> nets_;
	std::vector<std::vector<std::shared_ptr<Tensor<dtype>>>> learnable_params_;
	std::vector<std::vector<std::shared_ptr<Tensor<dtype>>>> momentum_params_;
	//std::vector<std::shared_ptr<Net<dtype>>> nets_;
	//std::vector<std::shared_ptr<Tensor<dtype>>> 
};

}




#endif
