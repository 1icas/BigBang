#include "../../include/util/math_function_ptr.h"

//#include <cuda_runtime.h>
//#include <device_launch_parameters.h>
//
//#include "cublas_v2.h"

#include "../../include/config.h"
#include "../../include/util/common.h"

template<typename dtype>
__global__ void gpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		c[index] = a[index] - alpha*b[index];
	}
}

template<typename dtype>
__global__ void gpu_column_sum_plus(const dtype* a, const int row, 
	const int column, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < column) {
		for (int i = 0; i < row; ++i) {
			b[index] += a[i*column + index];
		}
	}
}

template<typename dtype>
__global__ void gpu_mmadd(const dtype* a, const dtype* b,
	const int size, dtype* result) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		result[index] = a[index] + b[index];
	}
}

template<typename dtype>
__global__ void gpu_gen_git_label(const dtype* a, const int size, const int classes, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		b[index*classes + static_cast<int>(a[index] + 0.1)] = 1.;
	}
}

template<typename dtype>
__global__ void gpu_argmax(const dtype* a, const int row, const int column, dtype* b) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < row) {
		dtype t = a[index*column];
		int m = 0;
		for (int k = 1; k < column; ++k) {
			if (t < a[index*column + k]) {
				t = a[index*column + k];
				m = k;
			}
		}
		b[index] = m;
	}
}

template<typename dtype>
__global__ void gpu_equal_count(const dtype* a, const dtype* b, const int size, int* count) {
	const int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size) {
		if (a[index] == b[index]) atomicAdd(count, 1);
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
	cublasSgemm(Config::Get().CublasHandle(), op_b, op_a, n, m, k, &alpha, b, ldb,
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
	const int lda = trans_a ? m : k;
	const int ldb = trans_b ? k : n;
	cublasOperation_t op_a = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasOperation_t op_b = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
	cublasDgemm(Config::Get().CublasHandle(), op_b, op_a, n, m, k, &alpha, b, ldb,
		a, lda, &beta, c, n);
}

template<typename dtype>
void bigbang_gpu_column_sum_plus(const dtype* a, const int row, const int column, dtype* b) {
	cudaMemset(b, 0, sizeof(dtype)*column);
	gpu_column_sum_plus << <BigBangGetBlocks(column), THREAD_MAX_NUMS >> > (a, row, column, b);
}
template void bigbang_gpu_column_sum_plus<float>(const float* a, const int row, const int column, float* b);
template void bigbang_gpu_column_sum_plus<double>(const double* a, const int row, const int column, double* b);


template<typename dtype>
void bigbang_gpu_mmadd(const dtype* a, const dtype* b, 
	const int size, dtype* result) {
	gpu_mmadd << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> > (a, b, size, result);
}
template void bigbang_gpu_mmadd<float>(const float* a, const float* b,
	const int size, float* result);
template void bigbang_gpu_mmadd<double>(const double* a, const double* b,
	const int size, double* result);

template<typename dtype>
void bigbang_gpu_argmax(const dtype* a, const int row, const int column, dtype* b) {
	gpu_argmax << <BigBangGetBlocks(row), THREAD_MAX_NUMS >> > (a, row, column, b);
}
template void bigbang_gpu_argmax<float>(const float* a, const int row, const int column, float* b);
template void bigbang_gpu_argmax<double>(const double* a, const int row, const int column, double* b);

template<typename dtype>
void bigbang_gpu_equals_count(const dtype* a, const dtype* b, const int size, int* count) {
	gpu_equal_count << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> >(a, b, size, count) ;
}
template void bigbang_gpu_equals_count<float>(const float* a, const float* b, const int size, int* count);
template void bigbang_gpu_equals_count<double>(const double* a, const double* b, const int size, int* count);

template<typename dtype>
void bigbang_gpu_gen_fit_label(const dtype* a, const int size, const int classes, dtype* b) {
	gpu_gen_git_label << <BigBangGetBlocks(size), THREAD_MAX_NUMS >> > (a, size, classes, b);
}
template void bigbang_gpu_gen_fit_label<float>(const float* a, const int size, const int classes, float* b);
template void bigbang_gpu_gen_fit_label<double>(const double* a, const int size, const int classes, double* b);

void bigbang_gpu_random_uniform(const int size, unsigned int* output) {
	curandGenerate(Config::Get().CurandGenerator(), output, size);
}

}
