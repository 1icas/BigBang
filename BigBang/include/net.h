#ifndef NET_H
#define NET_H
#include <memory>
#include <vector>

#include "layer.h"
#include "layer_params_manage.h"
#include "tensor.h"

namespace BigBang {
template<typename dtype>
class Net {
public:
	explicit Net(const std::vector<LayerParamsManage<dtype>>& manages) :
		manages_(manages) {

	}

	void TrainDebug();

private:
	void GenerateLayers();
	void GenerateInput();


private:
	std::vector<LayerParamsManage<dtype>> manages_;
	std::vector<std::shared_ptr<Layer<dtype>>> layers_;
	std::vector<std::shared_ptr<Tensor<dtype>>> input_;

};
}



#endif
