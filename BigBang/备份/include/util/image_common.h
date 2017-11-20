#ifndef IMAGE_COMMON_H
#define IMAGE_COMMON_H

#include "../tensor.h"

namespace BigBang {
template <typename dtype>
double accuracy(const Tensor<dtype>* result, const Tensor<dtype>* predict);
}









#endif