#ifndef COMMON_H
#define COMMON_H

#include <cassert>
#include <cuda_runtime.h>
#include "../gtest.h"

#define bigbangcpumemset(s, ch, n) memset(s, ch, n)

#define NO_GPU \
	assert(false);

#ifndef CPU_ONLY

#define THREAD_MAX_NUMS 1024
#define bigbanggpumemset(s, ch, n) cudaMemset(s, ch, n);
#define CUDA_CHECK(state) \
	CHECK_EQ(state, cudaSuccess)

#endif


namespace BigBang {
#ifndef CPU_ONLY
inline int BigBangGetBlocks(const int n) {
	return (THREAD_MAX_NUMS + n + 1) / THREAD_MAX_NUMS;
}
#endif
}


#endif