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
//template <typename dtype>
//void bigbang_cpu_gemm(const dtype* a, const int a_row, const int a_column, bool transpose_a, 
//	const dtype* b, const int b_row, const int b_column, bool transpose_b, 
//	double alpha, const dtype* c, const int c_row, const int c_column, bool transpose_c, dtype* d) {
//	dtype* d_out = d;
//	dtype* _a = const_cast<dtype*>(a);
//	dtype* _b = const_cast<dtype*>(b);
//	dtype* _c = const_cast<dtype*>(c);
//
//	int _a_row = a_row;
//	int _a_column = a_column;
//	int _b_row = b_row;
//	int _b_column = b_column;
//	int _c_row = c_row;
//	int _c_column = c_column;
//
//	int dtype_size = sizeof(dtype);
//	if (transpose_a) {
//		_a = static_cast<dtype*>(malloc(dtype_size*a_row*a_column));
//		transpose(a, a_row, a_column, _a);
//		std::swap(_a_row, _a_column);
//	}
//	if (transpose_b) {
//		_b = static_cast<dtype*>(malloc(dtype_size*b_row*b_column));
//		transpose(b, b_row, b_column, _b);
//		std::swap(_b_row, _b_column);
//	}
//	if (transpose_c) {
//		_c = static_cast<dtype*>(malloc(dtype_size*c_row*c_column));
//		transpose(c, c_row, c_column, _c);
//		std::swap(_c_row, _c_column);
//	}
//
//	for (int i = 0; i < _a_row; ++i) {
//		for (int k = 0; k < _b_column; ++k) {
//			for (int m = 0; m < _a_column; ++m) {
//				*d_out += _a[i*_a_column + m] * _b[m*_b_column + k];
//			}
//			*d_out *= alpha;
//			if (_c) *d_out += _c[k];
//			++d_out;
//		}
//	}
//
//	if (transpose_a) free(_a);
//	if (transpose_b) free(_b);
//	if (transpose_c) free(_c);
//}
//
//template void bigbang_cpu_gemm<float>(const float* a, const int a_row, const int a_column, bool transpose_a, 
//	const float* b, const int b_row, const int b_column, bool transpose_b, double alpha, 
//	const float* c, const int c_row, const int c_column, bool transpose_c, float* d);
//template void bigbang_cpu_gemm<double>(const double* a, const int a_row, const int a_column, 
//	bool transpose_a, const double* b, const int b_row, const int b_column,
//	bool transpose_b, double alpha, const double* c, const int c_row, const int c_column, 
//	bool transpose_c, double* d);

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
	dtype* c) {
	dtype* _a = const_cast<dtype*>(a);
	dtype* _b = const_cast<dtype*>(b);

	int _a_row = m;
	int _a_column = k;
	int _b_row = k;
	int _b_column = n;

	int dtype_size = sizeof(dtype);
	if (trans_a) {
		_a = static_cast<dtype*>(malloc(dtype_size*_a_row*_a_column));
		transpose(a, _a_column, _a_row, _a);
	}
	if (trans_b) {
		_b = static_cast<dtype*>(malloc(dtype_size*_b_row*_b_column));
		transpose(b, _b_column, _b_row, _b);
	}

	for (int i = 0; i < _a_row; ++i) {
		for (int k = 0; k < _b_column; ++k) {
			dtype v = 0;
			for (int m = 0; m < _a_column; ++m) {
				v += _a[i*_a_column + m] * _b[m*_b_column + k];
			}
			v *= alpha;
			*c = v + beta * (*c);
			++c;
		}
	}

	if (trans_a) free(_a);
	if (trans_b) free(_b);
}

template void bigbang_cpu_gemm<float>(
	bool trans_a,
	bool trans_b,
	int m,
	int n,
	int k,
	const float alpha,
	const float* a,
	const float* b,
	const float beta,
	float* c);
template void bigbang_cpu_gemm<double>(
	bool trans_a,
	bool trans_b,
	int m,
	int n,
	int k,
	const double alpha,
	const double* a,
	const double* b,
	const double beta,
	double* c);


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
void bigbang_cpu_column_sum_plus(const dtype* a, const int row, const int column, dtype* b) {
	bigbangcpumemset(b, 0, sizeof(dtype)*column);
	for (int i = 0; i < column; ++i) {
		for (int k = 0; k < row; ++k) {
			b[i] += a[k*column + i];
		}
	}
}
template void bigbang_cpu_column_sum_plus<float>(const float* a, const int row, const int column, float* b);
template void bigbang_cpu_column_sum_plus<double>(const double* a, const int row, const int column, double* b);

template <typename dtype>
void row_sum_plus(const dtype* a, const int row, const int column, dtype* b) {
	bigbangcpumemset(b, 0, sizeof(dtype)*row);
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
	std::normal_distribution<dtype> urd(mean, std);
	for (int i = 0; i < size; ++i) {
		b[i] = urd(e);
	}
	/*std::knuth_b kb;
	std::normal_distribution<dtype> nd;
	for (int i = 0; i < size; ++i) {
		b[i] = nd(kb);
	}*/
}
template void GaussianDistribution<float>(const float mean, const float std, const int size, float* b);
template void GaussianDistribution<double>(const double mean, const double std, const int size, double* b);

template<typename dtype>
void bigbang_cpu_argmax(const dtype* a, const int row, const int column, dtype* b) {
	for (int i = 0; i < row; ++i) {
		dtype t = a[i*column];
		int m = 0;
		for (int k = 1; k < column; ++k) {
			if (t < a[i*column + k]) {
				t = a[i*column + k];
				m = k;
			}
		}
		b[i] = m;
	}
}
template void bigbang_cpu_argmax<float>(const float* a, const int row, const int column, float* b);
template void bigbang_cpu_argmax<double>(const double* a, const int row, const int column, double* b);

template<typename dtype>
void bigbang_cpu_equals_percent(const dtype* a, const dtype* b, const int size, dtype* percent) {
	int c = 0;
	for (int i = 0; i < size; ++i) {
		if (a[i] == b[i]) ++c;
	}
	*percent = c / size;
}
template void bigbang_cpu_equals_percent<float>(const float* a, const float* b, const int size, float* percent);
template void bigbang_cpu_equals_percent<double>(const double* a, const double* b, const int size, double* percent);

template<typename dtype>
void bigbang_cpu_equals_count(const dtype* a, const dtype* b, const int size, int* count) {
	int c = 0;
	for (int i = 0; i < size; ++i) {
		if (a[i] == b[i]) ++c;
	}
	*count = c;
}
template void bigbang_cpu_equals_count<float>(const float* a, const float* b, const int size, int* count);
template void bigbang_cpu_equals_count<double>(const double* a, const double* b, const int size, int* count);


template<typename dtype>
void bigbang_cpu_gen_fit_label(const dtype* a, const int size, const int classes, dtype* b) {
	for (int i = 0; i < size; ++i) {
		//TODO£º
		b[i*classes + static_cast<int>(a[i] + 0.1)] = 1;
	}
}
template void bigbang_cpu_gen_fit_label<float>(const float* a, const int size, const int classes, float* b);
template void bigbang_cpu_gen_fit_label<double>(const double* a, const int size, const int classes, double* b);

template<typename dtype>
void bigbang_cpu_random_uniform(const int n, const dtype a, const dtype b, dtype* c) {
	std::uniform_real_distribution<dtype> distribution(a, b);
	std::mt19937 mt(GenerateSeed());
	for (int i = 0; i < n; ++i) {
		c[i] = distribution(mt);
	}
}
template void bigbang_cpu_random_uniform<float>(const int n, const float a, const float b, float* c);
template void bigbang_cpu_random_uniform<double>(const int n, const double a, const double b, double* c);

template<typename dtype>
void bigbang_cpu_random_bernoulli(const int size, const dtype ratio, dtype* output) {
	std::bernoulli_distribution bd(ratio);
	std::mt19937 mt(GenerateSeed());
	for (int i = 0; i < size; ++i) {
		output[i] = bd(mt);
	}
}
template void bigbang_cpu_random_bernoulli<float>(const int size, const float ratio, float* output);
template void bigbang_cpu_random_bernoulli<double>(const int size, const double ratio, double* output);


template<typename dtype>
void bigbang_cpu_plus(const dtype* a, const int size, const dtype alpha, const dtype beta, dtype* b) {
	for (int i = 0; i < size; ++i) {
		b[i] = beta*b[i] + alpha*a[i];
	}
}
template void bigbang_cpu_plus<float>(const float* a, const int size, const float alpha, const float beta, float* b);
template void bigbang_cpu_plus<double>(const double* a, const int size, const double alpha, const double beta, double* b);

}