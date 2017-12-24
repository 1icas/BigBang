#ifndef MATH_FUNCTION_PTR_H
#define MATH_FUNCTION_PTR_H

#include <random>
#include <sys/timeb.h>

namespace BigBang {
//d = alpha* a*b + c;
	//TODO: need to refactor the function,  
	//need to the same as the gpu_gemm
//template<typename dtype>
//void bigbang_cpu_gemm(const dtype* a, const int a_row, const int a_column, bool transpose_a, const dtype* b, const int b_row, const int b_column,
//	bool transpose_b, double alpha, const dtype* c, const int c_row, const int c_column, bool transpose_c, dtype* d);

template<typename dtype>
void bigbang_cpu_gemm(
	bool trans_a,
	bool trans_b,
	int m,
	int n,
	int k,
	const dtype alpha,
	const dtype* a,
	const dtype* b,
	const dtype beta,
	dtype* c);

template<typename dtype>
void plus(const dtype* a, const int size, const dtype m, dtype* b);

template<typename dtype>
void bigbang_cpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c);

template<typename dtype>
void bigbang_cpu_plus(const dtype* a, const int size, const dtype alpha, const dtype beta, dtype* b);

template<typename dtype>
void bigbang_cpu_column_sum_plus(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void row_sum_plus(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void GaussianDistribution(const dtype mean, const dtype std, const int size, dtype* b);

template<typename dtype>
void bigbang_cpu_gen_fit_label(const dtype* a, const int size, const int classes, dtype* b);

template<typename dtype>
void bigbang_cpu_argmax(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void bigbang_cpu_equals_percent(const dtype* a, const dtype* b, const int size, dtype* percent);

template<typename dtype>
void bigbang_cpu_equals_count(const dtype* a, const dtype* b, const int size, int* count);

template<typename dtype>
void bigbang_cpu_random_uniform(const int n, const dtype a, const dtype b, dtype* c);

template<typename dtype>
void bigbang_cpu_random_bernoulli(const int size, const dtype ratio, dtype* output);

#ifndef CPU_ONLY

template<typename dtype>
void bigbang_gpu_gemm(
	bool trans_a,
	bool trans_b,
	int m,
	int n,
	int k,
	const dtype alpha,
	const dtype* a,
	const dtype* b,
	const dtype beta,
	dtype* c);

template<typename dtype>
void bigbang_gpu_column_sum_plus(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void bigbang_gpu_mmadd(
	const dtype* a,
	const dtype* b,
	const int size,
	dtype* result
);

template<typename dtype>
void bigbang_gpu_argmax(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void bigbang_gpu_equals_count(const dtype* a, const dtype* b, const int size, int* count);

template<typename dtype>
void bigbang_gpu_gen_fit_label(const dtype* a, const int size, const int classes, dtype* b);

template<typename dtype>
void bigbang_gpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c);

template<typename dtype>
void bigbang_gpu_plus(const dtype* a, const int size, const dtype alpha, const dtype beta, dtype* b);

void bigbang_gpu_random_uniform(const int size, unsigned int* output);

#endif

}




#endif
