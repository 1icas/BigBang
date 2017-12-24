#include "../../include/layers/softmax_cost_layer.h"
#include "../../include/layer_factory.h"

template<typename dtype>
void find_max_value1(const dtype* input, const int size, const int nums, dtype* output) {
	const int single_size = size / nums;
	for (int i = 0; i < nums; ++i) {
		dtype max = input[i*single_size + 0];
		for (int k = 1; k < single_size; ++k) {
			if (input[i*single_size + k] > max) {
				max = input[i*single_size + k];
			}
		}
		output[i] = max;
	}
}

namespace BigBang {

template<typename dtype>
void SoftmaxCostLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
	top->Reshape(std::vector<int>{bottom->shape(0), 1, 1, 1});
	softmax_result_.reset(new Tensor<dtype>(bottom->shape()));
	softmax_sum_.reset(new Tensor<dtype>(std::vector<int>{bottom->shape(0), 1, 1, 1}));
	max_num_.reset(new Tensor<dtype>(std::vector<int>{bottom->shape(0), 1, 1, 1}));
}

template<typename dtype>
void SoftmaxCostLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
	const dtype* bottom_data = bottom->cpu_data();
	const int nums = bottom->shape(0);
	const int size = bottom->size();
	const int per_data_size = size / nums;
	dtype* softmax_sum_data = softmax_sum_->mutable_cpu_data();
	dtype* softmax_result_data = softmax_result_->mutable_cpu_data();
	dtype* mutable_max_data = max_num_->mutable_cpu_data();
	dtype sum = 0;

	find_max_value1(bottom_data, size, nums, mutable_max_data);
	for (int i = 0; i < size; ++i) {
		if (i != 0 && i % per_data_size == 0) {
			softmax_sum_data[i / per_data_size - 1] = sum;
			sum = 0;
		}
		dtype r = exp(bottom_data[i] - mutable_max_data[i / per_data_size]);
		sum += r;
		softmax_result_data[i] = r;
	}
	softmax_sum_data[nums-1] = sum;

	for (int i = 0; i < size; ++i) {
		const int index = i / per_data_size;
		softmax_result_data[i] /= softmax_sum_data[index];
	}

	//compute the cost
	if (++count_ % times_ == 0) {
		const dtype* labels = top->cpu_data();
		dtype loss = 0;
		for (int i = 0; i < nums; ++i) {
			//solve the numerical issues
			loss += -log(std::max<dtype>(softmax_result_data[static_cast<int>(labels[i] + 0.1) + i*per_data_size], FLT_MIN));
		}
		std::cout << "softmax loglikelihood training " << count_ << " times, the error is: " << loss / nums << std::endl;
	}
	
}

template<typename dtype>
void SoftmaxCostLayer<dtype>::Backward_CPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
	const int size = bottom->size();
	const int nums = bottom->shape(0);
	const int internal_loop = size / nums;
	const dtype* top_data = top->cpu_data();
	const dtype* result_data = softmax_result_->cpu_data();
	dtype* bottom_diff = bottom->mutable_cpu_diff_data();
	for (int i = 0; i < nums; ++i) {
		const int label = top_data[i];
		for (int k = 0; k < internal_loop; ++k) {
			const int index = i*internal_loop + k;
			if (label == k) {
				bottom_diff[index] = result_data[index] - 1;
			}
			else {
				bottom_diff[index] = result_data[index];
			}
		}
	}
}

INSTANTIATE_CLASS(SoftmaxCostLayer);
REGISTRY_LAYER(SoftmaxCost);
}