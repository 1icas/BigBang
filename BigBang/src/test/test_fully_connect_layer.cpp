#include "test.h"


//template<typename dtype>
//void GetOptionalMatrix(const Tensor<dtype>* src, const int start_row, const int end_row, Tensor<dtype>* dest) {
//	const dtype* s_data = src->gpu_data();
//	dtype* d_data = dest->mutable_gpu_data();
//	const int column = src->shape(2)*src->shape(3);
//
//	for (int i = 0; i < (end_row - start_row); ++i) {
//		for (int j = 0; j < column; ++j) {
//			d_data[i*column + j] = s_data[(i + start_row)*column + j];
//		}
//	}
//	cudaMemcpy_s
//}

template<typename dtype>
void GetOptionalMatrix(const Tensor<dtype>* src, const int start_row, const int end_row, Tensor<dtype>* dest) {
	const dtype* s_data = src->cpu_data();
	dtype* d_data = dest->mutable_cpu_data();
	const int column = src->shape(2)*src->shape(3);
	/*s_data += column*start_row;
	cudaMemcpy(d_data, s_data, column*(end_row - start_row));
	*/

	for (int i = 0; i < (end_row - start_row); ++i) {
		for (int j = 0; j < column; ++j) {
			d_data[i*column + j] = s_data[(i + start_row)*column + j];
		}
	}
}


template<typename dtype>
void Test::TestFullyConnectLayer() {
	const int w = 28;
	const int h = 28;
	const int batch_size = 10;
	const int train_nums = 50000;
	const int test_nums = 10000;
	const int hidden_neuron_nums = 30;
	const int output_neuron_nums = 10;
	bool use_gpu = true;
	DataParams data_params(DataParams::DataSet::Mnist, 1, h, w, "./src/test/mnist_data/train-images.idx3-ubyte",
		"./src/test/mnist_data/train-labels.idx1-ubyte", 0, train_nums, false, "", "", 0, 0, true,
		"./src/test/mnist_data/t10k-images.idx3-ubyte", "./src/test/mnist_data/t10k-labels.idx1-ubyte",
		0, test_nums);
	DataReader<dtype> dr(data_params);
	dr.Read();
	std::shared_ptr<Tensor<dtype>> train_data_image = dr.GetTrainDataImage();
	std::shared_ptr<Tensor<dtype>> train_data_result = dr.GetTrainDataLabel();
	std::shared_ptr<Tensor<dtype>> test_data_image = dr.GetTestDataImage();
	std::shared_ptr<Tensor<dtype>> test_data_result = dr.GetTestDataLabel();

	//construct the layer
	std::shared_ptr<Tensor<dtype>> weights(new Tensor<dtype>(std::vector<int>{1, 1, w*h, hidden_neuron_nums}));
	std::shared_ptr<Tensor<dtype>> biases(new Tensor<dtype>(std::vector<int>{1, 1, hidden_neuron_nums, 1}));
	InnerProductLayerParams<dtype> params(3.0, 3.0, true, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
		0, 1), FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION, -1, 1), weights, biases);
	LayerParamsManage<dtype> manage;
	manage.use_gpu_ = use_gpu;
	manage.type_ = "InnerProduct";
	manage.inner_product_layer_params_ = params;
	InnerProductLayer<dtype> layer(manage);

	LayerParamsManage<dtype> manage1;
	manage1.use_gpu_ = use_gpu;
	manage1.type_ = "Sigmoid";
	SigmoidLayer<dtype> sl(manage1);

	std::shared_ptr<Tensor<dtype>> weights1(new Tensor<dtype>(std::vector<int>{1, 1, hidden_neuron_nums, output_neuron_nums}));
	std::shared_ptr<Tensor<dtype>> biases1(new Tensor<dtype>(std::vector<int>{1, 1, output_neuron_nums, 1}));
	InnerProductLayerParams<dtype> params1(3.0, 3.0, true, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
		0, 1), FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION, -1, 1), weights1, biases1);
	LayerParamsManage<dtype> manage2;
	manage2.use_gpu_ = use_gpu;
	manage2.type_ = "InnerProduct";
	manage2.inner_product_layer_params_ = params1;
	InnerProductLayer<dtype> layer1(manage2);

	LayerParamsManage<dtype> manage3;
	manage3.use_gpu_ = use_gpu;
	manage3.type_ = "Sigmoid";
	SigmoidLayer<dtype> sl1(manage3);
	
	CostFuncLayerParams<dtype> cflp();
	LayerParamsManage<dtype> manage4;
	manage4.use_gpu_ = use_gpu;
	manage4.type_ = "MSE";
	MSELayer<dtype> msel(manage4);

	Tensor<dtype> input(std::vector<int>{batch_size, 1, 1, h*w});
	Tensor<dtype> hidden(std::vector<int>{batch_size, 1, 1, hidden_neuron_nums});
	Tensor<dtype> output(std::vector<int>{batch_size, 1, 1, output_neuron_nums});
	Tensor<dtype> result(std::vector<int>{batch_size, 1, 1, output_neuron_nums});

	Tensor<dtype> hidden_test(std::vector<int>{test_nums, 1, 1, hidden_neuron_nums});
	Tensor<dtype> output_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});

	for (int l = 0; l < 30; ++l) {
		layer.SetUp(&input, &hidden);
		sl.SetUp(&hidden, &hidden);
		layer1.SetUp(&hidden, &output);
		msel.SetUp(&output, &result);
		
		for (int i = 0; i < 5000; ++i) {
			
			/*Tensor<dtype> input(std::vector<int>{batch_size, 1, 1, h*w});
			Tensor<dtype> hidden(std::vector<int>{batch_size, 1, 1, hidden_neuron_nums});
			Tensor<dtype> output(std::vector<int>{batch_size, 1, 1, output_neuron_nums});
			Tensor<dtype> result(std::vector<int>{batch_size, 1, 1, output_neuron_nums});*/

			int offset = batch_size*i*w*h;
			input.shared_data(*(train_data_image.get()));
			input.set_data_offset(offset);
			result.shared_data(*(train_data_result.get()));
			result.set_data_offset(batch_size*i*10);

			/*Tensor<dtype> input(std::vector<int>{batch_size, 1, 1, h*w});
			Tensor<dtype> hidden(std::vector<int>{batch_size, 1, 1, hidden_neuron_nums});
			Tensor<dtype> output(std::vector<int>{batch_size, 1, 1, output_neuron_nums});
			Tensor<dtype> result(std::vector<int>{batch_size, 1, 1, output_neuron_nums});

			GetOptionalMatrix<dtype>(train_data_image.get(), batch_size * i, batch_size*(i + 1), &input);
			GetOptionalMatrix<dtype>(train_data_result.get(), batch_size * i, batch_size*(i + 1), &result);

			layer.SetUp(&input, &hidden);
			sl.SetUp(&hidden, &hidden);
			layer1.SetUp(&hidden, &output);
			msel.SetUp(&output, &result);*/
			
			//layer1.printWeights();

			layer.Forward(&input, &hidden);

			sl.Forward(&hidden, &hidden);

			layer1.Forward(&hidden, &output);
	
			sl.Forward(&output, &output);

			//msel.Forward(&output, &result);
			msel.Backward(&result, &output);

			sl.Backward(&output, &output);
		
			layer1.Backward(&output, &hidden);

			sl.Backward(&hidden, &hidden);
		
			layer.Backward(&hidden, &input);
	
		//	layer1.printWeights();

			//double accur = accuracy(&result, &output);
			//std::cout << accur << std::endl;
		}

		//layer1.printWeights();
		//layer.printWeights();
		layer.SetUp(test_data_image.get(), &hidden_test);
		sl.SetUp(&hidden_test, &hidden_test);
		layer1.SetUp(&hidden_test, &output_test);

		layer.Forward(test_data_image.get(), &hidden_test); 
		sl.Forward(&hidden_test, &hidden_test);

		layer1.Forward(&hidden_test, &output_test);

		sl.Forward(&output_test, &output_test);

		double accur = accuracy(test_data_result.get(), &output_test);
		std::cout << accur << std::endl;
	}
}


template void Test::TestFullyConnectLayer<float>();
template void Test::TestFullyConnectLayer<double>();