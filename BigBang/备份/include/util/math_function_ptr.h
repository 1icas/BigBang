#ifndef MATH_FUNCTION_PTR_H
#define MATH_FUNCTION_PTR_H

#include <random>
#include <sys/timeb.h>

namespace BigBang {
//d = alpha* a*b + c;
	//TODO: need to refactor the function,  
	//need to the same as the gpu_gemm
template<typename dtype>
void bigbang_cpu_gemm(const dtype* a, const int a_row, const int a_column, bool transpose_a, const dtype* b, const int b_row, const int b_column,
	bool transpose_b, double alpha, const dtype* c, const int c_row, const int c_column, bool transpose_c, dtype* d);

template<typename dtype>
void plus(const dtype* a, const int size, const dtype m, dtype* b);

template<typename dtype>
void bigbang_cpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c);

template<typename dtype>
void bigbang_gpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c);

template<typename dtype>
void bigbang_cpu_column_sum_plus(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void row_sum_plus(const dtype* a, const int row, const int column, dtype* b);

template<typename dtype>
void GaussianDistribution(const dtype mean, const dtype std, const int size, dtype* b);

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


#endif

}




#endif
