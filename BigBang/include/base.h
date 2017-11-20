#ifndef BASE_H
#define BASE_H

#include <cassert>

#define MAX_CAPACITY 2147483648

#define DISABLE_COPY_AND_ASSIGNMENT(classname) \
	classname(const classname& c) = delete; \
	classname& operator=(const classname& c) = delete;

#define INSTANTIATE_CLASS(classname) \
	template class classname<float>; \
	template class classname<double>;


#define INSTANTIATE_CLASS_GPU_FUNCTION(classname) \
	template void classname<float>::Forward_GPU(const Tensor<float>* bottom, Tensor<float>* top); \
	template void classname<double>::Forward_GPU(const Tensor<double>* bottom, Tensor<double>* top); \
	template void classname<float>::Backward_GPU(const Tensor<float>* top, Tensor<float>* bottom); \
	template void classname<double>::Backward_GPU(const Tensor<double>* top, Tensor<double>* bottom);

#endif