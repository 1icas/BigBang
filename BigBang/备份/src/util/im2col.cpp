#include "../../include/util/im2col.h"
#include <memory>
#include "../../include/util/common.h"

namespace BigBang {

inline bool is_a_ge_zero_and_a_lt_b(int a, int b) {
	return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}

template<typename dtype>
void bigbang_cpu_im2col(const dtype* in, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w, const int padding_h, const int padding_w,
	const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
	dtype* out) {
	const int size = height * width;
	const int output_h = (height - kernel_h + 2 * padding_h) / stride_h + 1;
	const int output_w = (width - kernel_w + 2 * padding_w) / stride_w + 1;
	//TODO: should be test
	bigbangcpumemset(out, 0, sizeof(dtype)*output_h*output_w*channels*kernel_h*kernel_w);
	dtype* out_p = out;
	auto isPadding = [](const int x, const int y) {return x < 0 || x >= y;};
	for (int channels_index = 0; channels_index < channels; ++channels_index) {
		for (int kernel_row = 0; kernel_row < kernel_h; ++kernel_row) {
			for (int kernel_column = 0; kernel_column < kernel_w; ++kernel_column) {
				for (int output_row = 0; output_row < output_h; ++output_row) {
					const int input_row = -padding_h + output_row * stride_h + kernel_row;
					for (int output_column = 0; output_column < output_w; ++output_column) {
						const int input_column = -padding_w + output_column * stride_w + kernel_column;
						if (isPadding(input_row, height) || isPadding(input_column, width)) {
							*out_p++ = 0;
						}
						else {
							*out_p++ = in[channels_index * size + input_row*height + input_column];
						}
					}
				}
			}
		}
	}
}

template void bigbang_cpu_im2col<double>(const double* in, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w, const int padding_h, const int padding_w,
	const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
	double* out);
template void bigbang_cpu_im2col<float>(const float* in, const int channels, const int height, const int width,
	const int kernel_h, const int kernel_w, const int padding_h, const int padding_w,
	const int stride_h, const int stride_w, const int dilation_h, const int dilation_w,
	float* out);

template <typename Dtype>
void bigbang_cpu_col2im(const Dtype* data_col, const int channels,
	const int height, const int width, const int kernel_h, const int kernel_w,
	const int pad_h, const int pad_w,
	const int stride_h, const int stride_w,
	const int dilation_h, const int dilation_w,
	Dtype* data_im) {
	bigbangcpumemset(data_im, 0, sizeof(Dtype) * width * height);
	const int output_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
	const int output_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
	const int channel_size = height * width;
	for (int channel = channels; channel--; data_im += channel_size) {
		for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
			for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
				int input_row = -pad_h + kernel_row;
				for (int output_rows = output_h; output_rows; output_rows--) {
					if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
						data_col += output_w;
					}
					else {
						int input_col = -pad_w + kernel_col;
						for (int output_col = output_w; output_col; output_col--) {
							if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
								data_im[input_row * width + input_col] += *data_col;
							}
							data_col++;
							input_col += stride_w;
						}
					}
					input_row += stride_h;
				}
			}
		}
	}
}

template
	void bigbang_cpu_col2im<float>(const float* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		float* data_im);

template
	void bigbang_cpu_col2im<double>(const double* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		double* data_im);
}






