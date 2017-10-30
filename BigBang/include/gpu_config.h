#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "base.h"

#ifndef CPU_ONLY


#define THREAD_MAX_NUMS 1024

namespace BigBang {
inline int BigBangGetBlocks(const int n) {
	return (THREAD_MAX_NUMS + n + 1) / THREAD_MAX_NUMS;
}

class GPUConfig {
public:

	DISABLE_COPY_AND_ASSIGNMENT(GPUConfig);

	static GPUConfig& Get() {
		static GPUConfig config;
		return config;
	}

	cublasHandle_t& CublasHandle() {
		return cublas_handle_;
	}

private:
	GPUConfig() {
		Init();
	};

	void Init() {
		if (!init) {
			//init the cuda
			int count;
			cudaGetDeviceCount(&count);
			CHECK_GT(count, 0);
			//TODO:now, we only the one gpu
			cudaSetDevice(0);

			//init the cublas
			cublasCreate(&cublas_handle_);
			init = true;
		}
	}

private:
	bool init = false;
	cublasHandle_t cublas_handle_;
};

}

#endif

#endif
