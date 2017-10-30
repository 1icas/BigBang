#include "../include/math_function_ptr.h"

//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//
//#include "cublas_v2.h"

#include "../include/gpu_config.h"

template<typename dtype>
__global__ void gpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		c[index] = a[index] - alpha*b[index];
	}
}

namespace BigBang {
template<typename dtype>
void bigbang_gpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c) {
	gpu_minus << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> > (a, b, size, alpha, c);
}

template void bigbang_gpu_minus<float>(const float* a, const float* b, const int size, const float alpha, float* c);
template void bigbang_gpu_minus<double>(const double* a, const double* b, const int size, const double alpha, double* c);

template<> 
void bigbang_gpu_gemm<float>(
	bool trans_a,
	bool trans_b,
	int m,
	int n,
	int k,
	const float alpha,
	const float* a,
	const float* b,
	const float beta,
	float* c) {
	const int lda = trans_a ? m : k;
	const int ldb = trans_b ? k : n;
	cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasSgemm(GPUConfig::Get().CublasHandle(), op_b, op_a, n, m, k, &alpha, b, ldb,
		a, lda, &beta, c, n);
}

template<>
void bigbang_gpu_gemm<double>(
	bool trans_a,
	bool trans_b,
	int m,
	int n,
	int k,
	const double alpha,
	const double* a,
	const double* b,
	const double beta,
	double* c) { 
	const int lda = trans_a ? k : m;
	const int ldb = trans_b ? n : k;
	cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasDgemm(GPUConfig::Get().CublasHandle(), op_a, op_b, n, m, k, &alpha, b, ldb,
		a, lda, &beta, c, n);
}


}
