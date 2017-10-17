#include "test.h"


void Test::TestTensor_CPU() {
	std::vector<int> shape = { 1,2,3 };
	Tensor<float> t1(shape);
	CHECK_EQ(t1.size(), 6);
	const int size = t1.size();
	//change the cpu data, and check the gpu data(equal)
	float* cpu_mutable_data = t1.mutable_cpu_data();
	for (int i = 0; i < size; ++i) {
		cpu_mutable_data[i] = i;
	}
	const float* gpu_data = t1.gpu_data();
	float data_copy[6] = { 0 };
	cudaMemcpy(&data_copy, gpu_data, size * 4, cudaMemcpyDeviceToHost);
	for (int i = 0; i < size; ++i) {
		CHECK_EQ(cpu_mutable_data[i], data_copy[i]);
	}
}