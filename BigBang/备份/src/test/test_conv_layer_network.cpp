#include "test.h"

template<typename dtype>
void Test::TestConvLayerNetwork() {
	const int w = 28;
	const int h = 28;
	const int batch_size = 10;
	const int train_nums = 50000;
	const int test_nums = 10000;
	const int hidden_neuron_nums = 30;
	const int output_neuron_nums = 10;
	const int kernel_groups = 3;
	const dtype alpha = 0.1;
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

	//conv layer
	std::shared_ptr<Tensor<dtype>> kernels(new Tensor<dtype>(std::vector<int>{kernel_groups, 1, 5, 5}));
	ConvLayerParams<dtype> conv_params(kernel_groups, 1, 5, 5, 1, 1, 0, 0, alpha, alpha, false, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
		0, 1), FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
			0, 1), kernels, nullptr);
	LayerParamsManage<dtype> conv_manage;
	conv_manage.use_gpu_ = use_gpu;
	conv_manage.type_ = "Conv";
	conv_manage.conv_layer_params_ = conv_params;
	ConvLayer<dtype> conv_layer(conv_manage);

	//max pooling layer
	PoolingLayerParams<dtype> pooling_params(PoolingLayerParams<dtype>::Pool::MaxPool, 2, 2, 2, 2);
	LayerParamsManage<dtype> pool_manage;
	pool_manage.use_gpu_ = use_gpu;
	pool_manage.type_ = "Pool";
	pool_manage.pooling_layer_params_ = pooling_params;
	PoolingLayer<dtype> pooling_layer(pool_manage);

	//construct the layer
	std::shared_ptr<Tensor<dtype>> weights(new Tensor<dtype>(std::vector<int>{1, 1, kernel_groups*12*12, hidden_neuron_nums}));
	std::shared_ptr<Tensor<dtype>> biases(new Tensor<dtype>(std::vector<int>{1, 1, hidden_neuron_nums, 1}));
	InnerProductLayerParams<dtype> params(alpha, alpha, true, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
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
	InnerProductLayerParams<dtype> params1(alpha, alpha, true, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
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

	Tensor<dtype> input(std::vector<int>{batch_size, 1, h, w});
	Tensor<dtype> conv_tensor(std::vector<int>{batch_size, kernel_groups, 24, 24});
	Tensor<dtype> pooling_tensor(std::vector<int>{batch_size, kernel_groups, 12, 12});
	Tensor<dtype> hidden(std::vector<int>{batch_size, 1, 1, hidden_neuron_nums});
	Tensor<dtype> output(std::vector<int>{batch_size, 1, 1, output_neuron_nums});
	Tensor<dtype> result(std::vector<int>{batch_size, 1, 1, output_neuron_nums});


	Tensor<dtype> conv_tensor_test(std::vector<int>{test_nums, kernel_groups, 24, 24});
	Tensor<dtype> pooling_tensor_test(std::vector<int>{test_nums, kernel_groups, 12, 12});
	Tensor<dtype> hidden_test(std::vector<int>{test_nums, 1, 1, hidden_neuron_nums});
	Tensor<dtype> output_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});
	Tensor<dtype> result_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});

	//Tensor<dtype> hidden_test(std::vector<int>{test_nums, 1, 1, hidden_neuron_nums});
	//Tensor<dtype> output_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});

	for (int l = 0; l < 100; ++l) {
		conv_layer.SetUp(&input, &conv_tensor);
		pooling_layer.SetUp(&conv_tensor, &pooling_tensor);
		layer.SetUp(&pooling_tensor, &hidden);
		sl.SetUp(&hidden, &hidden);
		layer1.SetUp(&hidden, &output);
		msel.SetUp(&output, &result);

		for (int i = 0; i < 5000; ++i) {

			int offset = batch_size*i*w*h;
			input.shared_data(*(train_data_image.get()));
			input.set_data_offset(offset);
			result.shared_data(*(train_data_result.get()));
			result.set_data_offset(batch_size* i * 10);

			conv_layer.Forward(&input, &conv_tensor);
			sl.Forward(&conv_tensor, &conv_tensor);
			pooling_layer.Forward(&conv_tensor, &pooling_tensor);

			layer.Forward(&pooling_tensor, &hidden);

			sl.Forward(&hidden, &hidden);

			layer1.Forward(&hidden, &output);

			sl.Forward(&output, &output);

			//msel.Forward(&output, &result);
			msel.Backward(&result, &output);

			sl.Backward(&output, &output);

			layer1.Backward(&output, &hidden);

			sl.Backward(&hidden, &hidden);

			layer.Backward(&hidden, &pooling_tensor);
				
			pooling_layer.Backward(&pooling_tensor, &conv_tensor);
			sl.Backward(&conv_tensor, &conv_tensor);

			conv_layer.Backward(&conv_tensor, &input);

			//	layer1.printWeights();

			//double accur = accuracy(&result, &output);
		//	std::cout << accur << std::endl;
		}

		//layer1.printWeights();
		//layer.printWeights();
		conv_layer.SetUp(test_data_image.get(), &conv_tensor_test);
		pooling_layer.SetUp(&conv_tensor_test, &pooling_tensor_test);
		layer.SetUp(&pooling_tensor_test, &hidden_test);
		sl.SetUp(&hidden_test, &hidden_test);
		layer1.SetUp(&hidden_test, &output_test);
		conv_layer.Forward(test_data_image.get(), &conv_tensor_test);
		sl.Forward(&conv_tensor_test, &conv_tensor_test);
		pooling_layer.Forward(&conv_tensor_test, &pooling_tensor_test);
		layer.Forward(&pooling_tensor_test, &hidden_test);
		sl.Forward(&hidden_test, &hidden_test);
		layer1.Forward(&hidden_test, &output_test);
		sl.Forward(&output_test, &output_test);

		double accur = accuracy(test_data_result.get(), &output_test);
		std::cout << accur << std::endl;
	/*	layer.SetUp(test_data_image.get(), &hidden_test);
		sl.SetUp(&hidden_test, &hidden_test);
		layer1.SetUp(&hidden_test, &output_test);

		layer.Forward(test_data_image.get(), &hidden_test);
		sl.Forward(&hidden_test, &hidden_test);

		layer1.Forward(&hidden_test, &output_test);

		sl.Forward(&output_test, &output_test);

		double accur = accuracy(test_data_result.get(), &output_test);
		std::cout << accur << std::endl;*/
	}
}

template void Test::TestConvLayerNetwork<float>();
template void Test::TestConvLayerNetwork<double>();




template<typename dtype>
void Test::TestConvLayerNetwork1() {
	const int w = 28;
	const int h = 28;
	const int batch_size = 10;
	const int train_nums = 50000;
	const int test_nums = 10000;
	const int hidden_neuron_nums = 100;
	const int output_neuron_nums = 10;
	const int kernel_groups = 3;
	const dtype alpha = 0.1;
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

	//conv layer
	std::shared_ptr<Tensor<dtype>> kernels(new Tensor<dtype>(std::vector<int>{kernel_groups, 1, 5, 5}));
	ConvLayerParams<dtype> conv_params(kernel_groups, 1, 5, 5, 1, 1, 0, 0, alpha, alpha, false, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
		0, 1), FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
			0, 1), kernels, nullptr);
	LayerParamsManage<dtype> conv_manage;
	conv_manage.use_gpu_ = false;//use_gpu;
	conv_manage.type_ = "Conv";
	conv_manage.conv_layer_params_ = conv_params;
	ConvLayer<dtype> conv_layer(conv_manage);

	//max pooling layer
	PoolingLayerParams<dtype> pooling_params(PoolingLayerParams<dtype>::Pool::MaxPool, 2, 2, 2, 2);
	LayerParamsManage<dtype> pool_manage;
	pool_manage.use_gpu_ = use_gpu;
	pool_manage.type_ = "Pool";
	pool_manage.pooling_layer_params_ = pooling_params;
	PoolingLayer<dtype> pooling_layer(pool_manage);

	//construct the layer
	std::shared_ptr<Tensor<dtype>> weights(new Tensor<dtype>(std::vector<int>{1, 1, kernel_groups * 12 * 12, hidden_neuron_nums}));
	std::shared_ptr<Tensor<dtype>> biases(new Tensor<dtype>(std::vector<int>{1, 1, hidden_neuron_nums, 1}));
	InnerProductLayerParams<dtype> params(alpha, alpha, true, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
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
	InnerProductLayerParams<dtype> params1(alpha, alpha, true, FillerParams<dtype>(FillerParams<dtype>::FillerType::GAUSSIAN_DISTRIBUTION,
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

	Tensor<dtype> input(std::vector<int>{batch_size, 1, h, w});
	Tensor<dtype> conv_tensor(std::vector<int>{batch_size, kernel_groups, 24, 24});
	Tensor<dtype> pooling_tensor(std::vector<int>{batch_size, kernel_groups, 12, 12});
	Tensor<dtype> hidden(std::vector<int>{batch_size, 1, 1, hidden_neuron_nums});
	Tensor<dtype> output(std::vector<int>{batch_size, 1, 1, output_neuron_nums});
	Tensor<dtype> result(std::vector<int>{batch_size, 1, 1, output_neuron_nums});


	Tensor<dtype> conv_tensor_test(std::vector<int>{test_nums, kernel_groups, 24, 24});
	Tensor<dtype> pooling_tensor_test(std::vector<int>{test_nums, kernel_groups, 12, 12});
	Tensor<dtype> hidden_test(std::vector<int>{test_nums, 1, 1, hidden_neuron_nums});
	Tensor<dtype> output_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});
	Tensor<dtype> result_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});

	//Tensor<dtype> hidden_test(std::vector<int>{test_nums, 1, 1, hidden_neuron_nums});
	//Tensor<dtype> output_test(std::vector<int>{test_nums, 1, 1, output_neuron_nums});

	for (int l = 0; l < 100; ++l) {
		conv_layer.SetUp(&input, &conv_tensor);
		pooling_layer.SetUp(&conv_tensor, &pooling_tensor);
		layer.SetUp(&pooling_tensor, &hidden);
		sl.SetUp(&hidden, &hidden);
		layer1.SetUp(&hidden, &output);
		msel.SetUp(&output, &result);

		for (int i = 0; i < 5000; ++i) {

			int offset = batch_size*i*w*h;
			input.shared_data(*(train_data_image.get()));
			input.set_data_offset(offset);
			result.shared_data(*(train_data_result.get()));
			result.set_data_offset(batch_size* i * 10);

			conv_layer.Forward(&input, &conv_tensor);
			sl.Forward(&conv_tensor, &conv_tensor);
			pooling_layer.Forward(&conv_tensor, &pooling_tensor);

			layer.Forward(&pooling_tensor, &hidden);

			sl.Forward(&hidden, &hidden);

			layer1.Forward(&hidden, &output);

			sl.Forward(&output, &output);

			//msel.Forward(&output, &result);
			msel.Backward(&result, &output);

			sl.Backward(&output, &output);

			layer1.Backward(&output, &hidden);

			sl.Backward(&hidden, &hidden);

			layer.Backward(&hidden, &pooling_tensor);

			pooling_layer.Backward(&pooling_tensor, &conv_tensor);
			sl.Backward(&conv_tensor, &conv_tensor);

			conv_layer.Backward(&conv_tensor, &input);

			//	layer1.printWeights();

			//double accur = accuracy(&result, &output);
			//	std::cout << accur << std::endl;
		}

		//layer1.printWeights();
		//layer.printWeights();
		conv_layer.SetUp(test_data_image.get(), &conv_tensor_test);
		pooling_layer.SetUp(&conv_tensor_test, &pooling_tensor_test);
		layer.SetUp(&pooling_tensor_test, &hidden_test);
		sl.SetUp(&hidden_test, &hidden_test);
		layer1.SetUp(&hidden_test, &output_test);
		conv_layer.Forward(test_data_image.get(), &conv_tensor_test);
		sl.Forward(&conv_tensor_test, &conv_tensor_test);
		pooling_layer.Forward(&conv_tensor_test, &pooling_tensor_test);
		layer.Forward(&pooling_tensor_test, &hidden_test);
		sl.Forward(&hidden_test, &hidden_test);
		layer1.Forward(&hidden_test, &output_test);
		sl.Forward(&output_test, &output_test);

		double accur = accuracy(test_data_result.get(), &output_test);
		std::cout << accur << std::endl;
		/*	layer.SetUp(test_data_image.get(), &hidden_test);
		sl.SetUp(&hidden_test, &hidden_test);
		layer1.SetUp(&hidden_test, &output_test);

		layer.Forward(test_data_image.get(), &hidden_test);
		sl.Forward(&hidden_test, &hidden_test);

		layer1.Forward(&hidden_test, &output_test);

		sl.Forward(&output_test, &output_test);

		double accur = accuracy(test_data_result.get(), &output_test);
		std::cout << accur << std::endl;*/
	}
}

template void Test::TestConvLayerNetwork1<float>();
template void Test::TestConvLayerNetwork1<double>();