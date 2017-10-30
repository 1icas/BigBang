#ifndef BASE_H
#define BASE_H

#include <cassert>

#define CHECK_OP(op, val1, val2) \
	if(!(val1 op val2)) assert(false);

#define CHECK_EQ(a, b) \
	CHECK_OP(==, a, b)

#define CHECK_GT(a, b) \
	CHECK_OP(>, a, b)

#define CHECK_LT(a, b) \
	CHECK_OP(<, a, b)

#define CHECK_GTE(a, b) \
	CHECK_OP(>=, a, b)

#define CHECK_LTE(a, b) \
	CHECK_OP(<=, a, b)

#define THROW_EXCEPTION assert(false)

#define VALIDATE_POINTER(p) \
	assert(p != nullptr);


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