#ifndef FILLER_H
#define FILLER_H

#include "tensor.h"

namespace BigBang {

template<typename dtype>
class Filler {
public:
	virtual void Fill(Tensor<dtype>* t) = 0;
};

}




#endif
