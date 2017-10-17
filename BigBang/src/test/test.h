#ifndef TEST_H
#define TEST_H

#include <vector>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "../../include/base.h"
#include "../../include/log.h"
#include "../../include/tensor.h"
using namespace BigBang;

class Test {
public:
	Test() {
		CUDA_CHECK(cudaSetDevice(0));
	}


	void TestAll() {
		TestTensor_CPU();
		TestTensor_GPU();
	}

private:
	void TestTensor_CPU();
	void TestTensor_GPU();
};

#endif
