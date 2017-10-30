#ifndef LOAD_DATA_H
#define LOAD_DATA_H

#include <iostream>
#include "../tensor.h"

namespace BigBang {

template<typename dtype>
void ReadMnistImage(const std::string& filename, const int start_data_index, const int end_data_index, 
	const int channels, const int h, const int w, Tensor<dtype>* m);
template<typename dtype>
void ReadMnistLabel(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<dtype>* m);

}



#endif
