#ifndef NET_H
#define NET_H
#include <memory>
#include <vector>

#include "layer.h"
#include "tensor.h"
#include "../proto/bigbang.pb.h"

namespace BigBang {
template<typename dtype>
class Net {
public:
	explicit Net(const NetParameter& params) :
		net_params_(params) {
		Initialize();
	}

	//void TrainDebug();
	void Train();

	int TestValidateData();

	std::vector<std::shared_ptr<Layer<dtype>>> Layers() const {
		return layers_;
	}

private:
	void Initialize();

private:
	NetParameter net_params_;
	std::vector<std::shared_ptr<Layer<dtype>>> layers_;
	std::vector<std::shared_ptr<Tensor<dtype>>> input_;
	std::shared_ptr<Tensor<dtype>> predicted_tensor_;

};
}



#endif
