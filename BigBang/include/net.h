#ifndef NET_H
#define NET_H
#include <memory>
#include <vector>

#include "layer.h"
#include "layer_params_manage.h"
#include "tensor.h"
#include "../proto/bigbang.pb.h"

namespace BigBang {
template<typename dtype>
class Net {
public:
	explicit Net(const NetParameter& params) :
		net_params_(params) {

	}

	void TrainDebug();

private:
	void GenerateLayers();
	void GenerateInput();

	void Initialize();



private:
	NetParameter net_params_;
	std::vector<std::shared_ptr<Layer<dtype>>> layers_;
	std::vector<std::shared_ptr<Tensor<dtype>>> input_;

};
}



#endif
