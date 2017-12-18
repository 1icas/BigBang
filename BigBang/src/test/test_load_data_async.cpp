#include "test.h"

void Test::TestLoadDataAsync() {
	LayerParameter l_p;
	auto data_layer_params = l_p.mutable_data_layer_param();
	data_layer_params->set_batch_size(10);
	data_layer_params->set_cache_batch_count(10);
	data_layer_params->set_source("D:/deeplearning/cifar_lmdb/cifar10_train.mdb");

	std::shared_ptr<DataLayer<double>> data_layer(new DataLayer<double>(l_p));
	Tensor<double>* bottom = new Tensor<double>(std::vector<int>{10, 3, 32, 32});
	Tensor<double>* top = new Tensor<double>(std::vector<int>{10, 3, 32, 32});
	data_layer->SetUp(bottom, top);
	data_layer->Forward(bottom, top);

}
