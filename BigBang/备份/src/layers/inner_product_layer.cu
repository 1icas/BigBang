//#include "../../include/layers/inner_product_layer.h"
//
//#include "../../include/util/math_function_ptr.h"
//
//namespace BigBang {
//
//template<typename dtype>
//void InnerProductLayer<dtype>::Forward_GPU(const Tensor<dtype>* bottom, Tensor<dtype>* top) {
//	bigbang_gpu_gemm<dtype>(false, false, bottom_row_, weights_column_, bottom_column_, 1.,
//		bottom->gpu_data(), weights_->gpu_data(), 0., top->mutable_gpu_data());
//}
//
//template<typename dtype>
//void InnerProductLayer<dtype>::Backward_GPU(const Tensor<dtype>* top, Tensor<dtype>* bottom) {
//	const dtype* bottom_data = bottom->gpu_data();
//	const dtype* top_diff_data = top->gpu_diff_data();
//	//get the delta
//	bigbang_gpu_gemm<dtype>(false, true, top_row_, weights_column, top_column_, 1., top_diff_data,
//		weights_->gpu_data(), 0, bottom->mutable_gpu_diff_data());
//	UpdateParams_GPU(bottom_data, top_diff_data);
//}
//
//template<typename dtype>
//void InnerProductLayer<dtype>::UpdateParams_GPU(const dtype* bottom_data, const dtype* delta) {
//	//update the biases
//	if (use_biases_) {
//		dtype* biases_mutable_diff_data = biases_->mutable_gpu_data();
//		bigbang_gpu_column_sum_plus(delta, bottom_row_, biases_row_, biases_mutable_diff_data);
//		bigbang_gpu_minus(biases_->gpu_data(), biases_mutable_diff_data, biases_row_, alpha_ / bottom_row_,
//			biases_->mutable_gpu_data());
//	}
//
//	//update the weights
//	dtype* weights_diff_data = weights_->mutable_gpu_diff_data();
//	cudaMemset(weights_diff_data, 0, sizeof(dtype)*weights_row_*weights_column_);
//	/*bigbang_cpu_gemm(bottom_data, bottom_row_, bottom_column_, true, delta, top_row_, top_column_,
//		false, alpha_ / bottom_row_, (dtype*)nullptr, 0, 0, false, weights_diff_data);*/
//	bigbang_gpu_gemm<dtype>(true, false, bottom_row_, top_column_, bottom_column_, alpha_ / bottom_row_,
//		bottom_data, delta, 0, weights_diff_data);
//	bigbang_gpu_minus(weights_->gpu_data(), weights_diff_data, weights_row_*weights_column_, 
//		static_cast<dtype>(1.0), weights_->mutable_gpu_data());
//}
//
//
//
//}