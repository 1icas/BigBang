#include "test.h"

__global__ void plus(float* r) {
	r[threadIdx.x] = 10;
}

void Test::TestTensor_GPU() {
	std::vector<int> shape = { 3,3 };
	Tensor<float> t1(shape);
	CHECK_EQ(t1.size(), 9);
	const int size = t1.size();
	//change the gpu data, and check the cpu data(equal)
	float* gpu_mutable_data = t1.mutable_gpu_data();
	plus <<<1, size >>> (gpu_mutable_data);
	CUDA_CHECK(cudaDeviceSynchronize());
	const float* cpu_data = t1.cpu_data();
	
	for (int i = 0; i < size; ++i) {
		CHECK_EQ(cpu_data[i], 10);
	}
}
