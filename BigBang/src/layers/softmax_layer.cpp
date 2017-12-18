#include "../../include/layers/softmax_layer.h"
#include "../../include/layer_factory.h"

template<typename dtype>
void find_max_value(const dtype* input, const int size, const int nums, dtype* output) {
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
	void SoftmaxLayer<dtype>::Prepare(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		CHECK_EQ(bottom->dimension(), DATA_DIMENSION);
		top->Reshape(bottom->shape());
		softmax_sum_.reset(new Tensor<dtype>(std::vector<int>{bottom->shape(0), 1, 1, 1}));
		max_num_.reset(new Tensor<dtype>(std::vector<int>{bottom->shape(0), 1, 1, 1}));
	}

	template<typename dtype>
	void SoftmaxLayer<dtype>::Forward_CPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
		const dtype* bottom_data = bottom->cpu_data();
		const int nums = bottom->shape(0);
		const int size = bottom->size();
		const int per_data_size = size / nums;
		dtype* mutable_softmax_sum_data = softmax_sum_->mutable_cpu_data();
		dtype* mutable_top_data = top->mutable_cpu_data();
		dtype* mutable_max_data = max_num_->mutable_cpu_data();
		dtype sum = 0;
		find_max_value(bottom_data, size, nums, mutable_max_data);
		for (int i = 0; i < size; ++i) {
			if (i != 0 && i % per_data_size == 0) {
				mutable_softmax_sum_data[i / per_data_size - 1] = sum;
				sum = 0;
			}
			//sovle the numerical issues(overflow, underflow)
			dtype r = exp(bottom_data[i] - mutable_max_data[i / per_data_size]);
			sum += r;
			mutable_top_data[i] = r;
		}
		mutable_softmax_sum_data[nums - 1] = sum;

		for (int i = 0; i < size; ++i) {
			const int index = i / per_data_size;
			mutable_top_data[i] /= mutable_softmax_sum_data[index];
		}
	}

	INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTRY_LAYER(Softmax);
}