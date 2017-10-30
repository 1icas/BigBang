#include "test.h"

void Test::TestGpuGemm() {
	const int m = 2;
	const int n = 3;
	const int k = 4;
	Tensor<float> a(std::vector<int>{m, k});
	Tensor<float> b(std::vector<int>{k, n});
	float* a_data = a.mutable_cpu_data();
	float* b_data = b.mutable_cpu_data();
	for (int i = 0; i < m*k; ++i) {
		a_data[i] = i;
	}
	for (int i = 0; i < k*n; ++i) {
		b_data[i] = 10 + i;
	}
	Tensor<float> c(std::vector<int>{m, n});
	Tensor<float> d(std::vector<int>{m, n});

	bigbang_cpu_gemm<float>(a.cpu_data(), m, k, false, b.cpu_data(), k, n, false, 1.0, nullptr, 0, 0, false, 
		c.mutable_cpu_data());
	float alpha = 1.f;
	float beta = 0.;
	bigbang_gpu_gemm<float>(false, false, m, n, k, 1., a.gpu_data(), b.gpu_data(), 0., d.mutable_gpu_data());
	const float* c_data = c.cpu_data();
	const float* d_data = d.cpu_data();
	for (int i = 0; i < m*n; ++i) {
		CHECK_EQ(c_data[i], d_data[i]);
	}
}