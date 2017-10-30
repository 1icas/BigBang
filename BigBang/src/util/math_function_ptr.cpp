#include "../../include/util/math_function_ptr.h"

#include <algorithm>

#include "../../include/util/common.h"

int GenerateSeed() {
	struct timeb timeSeed;
	ftime(&timeSeed);
	return timeSeed.time * 1000 + timeSeed.millitm;
}

namespace BigBang {
template<typename dtype>
void transpose(const dtype* a, const int row, const int column, dtype* b) {
	int size = row * column;
	for (int i = 0; i < size; ++i) {
		int k = i / row;
		int j = i % row;
		b[i] = a[column*j + k];
	}
}

//TODO: should we need to memset the d pointer?
template <typename dtype>
void bigbang_cpu_gemm(const dtype* a, const int a_row, const int a_column, bool transpose_a, 
	const dtype* b, const int b_row, const int b_column, bool transpose_b, 
	double alpha, const dtype* c, const int c_row, const int c_column, bool transpose_c, dtype* d) {
	dtype* d_out = d;
	dtype* _a = const_cast<dtype*>(a);
	dtype* _b = const_cast<dtype*>(b);
	dtype* _c = const_cast<dtype*>(c);

	int _a_row = a_row;
	int _a_column = a_column;
	int _b_row = b_row;
	int _b_column = b_column;
	int _c_row = c_row;
	int _c_column = c_column;

	int dtype_size = sizeof(dtype);
	if (transpose_a) {
		_a = static_cast<dtype*>(malloc(dtype_size*a_row*a_column));
		transpose(a, a_row, a_column, _a);
		std::swap(_a_row, _a_column);
	}
	if (transpose_b) {
		_b = static_cast<dtype*>(malloc(dtype_size*b_row*b_column));
		transpose(b, b_row, b_column, _b);
		std::swap(_b_row, _b_column);
	}
	if (transpose_c) {
		_c = static_cast<dtype*>(malloc(dtype_size*c_row*c_column));
		transpose(c, c_row, c_column, _c);
		std::swap(_c_row, _c_column);
	}

	for (int i = 0; i < _a_row; ++i) {
		for (int k = 0; k < _b_column; ++k) {
			for (int m = 0; m < _a_column; ++m) {
				*d_out += _a[i*_a_column + m] * _b[m*_b_column + k];
			}
			*d_out *= alpha;
			if (_c) *d_out += _c[k];
			++d_out;
		}
	}

	if (transpose_a) free(_a);
	if (transpose_b) free(_b);
	if (transpose_c) free(_c);
}
template void bigbang_cpu_gemm<float>(const float* a, const int a_row, const int a_column, bool transpose_a, 
	const float* b, const int b_row, const int b_column, bool transpose_b, double alpha, 
	const float* c, const int c_row, const int c_column, bool transpose_c, float* d);
template void bigbang_cpu_gemm<double>(const double* a, const int a_row, const int a_column, 
	bool transpose_a, const double* b, const int b_row, const int b_column,
	bool transpose_b, double alpha, const double* c, const int c_row, const int c_column, 
	bool transpose_c, double* d);

template <typename dtype>
void plus(const dtype* src, const int size, const dtype v, dtype* dst) {
	for (int i = 0; i < size; ++i) {
		dst[i] = src[i] + v;
	}
}
template void plus<float>(const float* a, const int size, const float m, float* b);
template void plus<double>(const double* a, const int size, const double m, double* b);

template <typename dtype>
void bigbang_cpu_minus(const dtype* a, const dtype* b, const int size, const dtype alpha, dtype* c) {
	for (int i = 0; i < size; ++i) {
		c[i] = a[i] - alpha*b[i];
	}
}
template void bigbang_cpu_minus<float>(const float* a, const float* b, const int size, const float alpha, float* c);
template void bigbang_cpu_minus<double>(const double* a, const double* b, const int size, const double alpha, double* c);

template <typename dtype>
void column_sum_plus(const dtype* a, const int row, const int column, dtype* b) {
	bigbangmemset(b, 0, sizeof(dtype)*column);
	for (int i = 0; i < column; ++i) {
		for (int k = 0; k < row; ++k) {
			b[i] += a[k*column + i];
		}
	}
}
template void column_sum_plus<float>(const float* a, const int row, const int column, float* b);
template void column_sum_plus<double>(const double* a, const int row, const int column, double* b);

template <typename dtype>
void row_sum_plus(const dtype* a, const int row, const int column, dtype* b) {
	bigbangmemset(b, 0, sizeof(dtype)*row);
	for (int i = 0; i < row; ++i) {
		for (int k = 0; k < column; ++k) {
			b[i] += a[i*column + k];
		}
	}
}
template void row_sum_plus<float>(const float* a, const int row, const int column, float* b);
template void row_sum_plus<double>(const double* a, const int row, const int column, double* b);

template<typename dtype>
void GaussianDistribution(const dtype mean, const dtype std, const int size, dtype* b) {
	std::default_random_engine e(GenerateSeed());
	std::uniform_real_distribution<dtype> urd(mean, std);
	for (int i = 0; i < size; ++i) {
		b[i] = urd(e);
	}
}
template void GaussianDistribution<float>(const float mean, const float std, const int size, float* b);
template void GaussianDistribution<double>(const double mean, const double std, const int size, double* b);

}