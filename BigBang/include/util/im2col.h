#ifndef IM2COL_H
#define IM2COL_H

namespace BigBang {


	//卷积核不做扩展
	template<typename dtype>
	void im2col(const dtype* in, const int channels, const int width, const int height,
		const int kernel_w, const int kernel_h, const int padding_w, const int padding_h,
		const int stride_w, const int stride_h, const int dilation_w, const int dilation_h,
		dtype* out);

	template <typename dtype>
	void col2im(const dtype* data_col, const int channels,
		const int height, const int width, const int kernel_h, const int kernel_w,
		const int pad_h, const int pad_w,
		const int stride_h, const int stride_w,
		const int dilation_h, const int dilation_w,
		dtype* data_im);
}


#endif