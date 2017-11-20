#include "../../include/util/image_common.h"

#include <iostream>

namespace BigBang {
template <typename dtype>
double accuracy(const Tensor<dtype>* result, const Tensor<dtype>* predict) {
	const int p_row = predict->shape(0);
	const int p_column = predict->shape(3);
	std::vector<int> right(p_row);
	const dtype* p_data = predict->cpu_data();
	//for (int i = 0; i < 10000; ++i) {
	//	std::cout << p_data[i] << std::endl;
	//}
	for (int i = 0; i < p_row; ++i) {
		double max = FLT_MIN;
		for (int k = 0; k < p_column; ++k) {
			if (p_data[i*p_column + k] > max) {
				max = p_data[i*p_column + k];
				right[i] = k;
			}
		}
	}
	int count = 0;
	const dtype* r_data = result->cpu_data();
	for (int i = 0; i < p_row; ++i) {
		if (abs(r_data[i*p_column + right[i]] - 1)< 0.1) {
			++count;
		}
	}
	return (double)count / (double)p_row;
}

template double accuracy<float>(const Tensor<float>* result, const Tensor<float>* predict);
template double accuracy<double>(const Tensor<double>* result, const Tensor<double>* predict);
}