#ifndef CONFIG_H_
#define CONFIG_H_

#include <cuda_runtime.h>
#include <curand.h>
#include "cublas_v2.h"

#include "base.h"
#include "gtest.h"



namespace BigBang {

class Config {
public:
	enum ProcessUnit {
		CPU,
		GPU
	};

	DISABLE_COPY_AND_ASSIGNMENT(Config);

	static Config& Get() {
		static Config config;
		return config;
	}

	ProcessUnit mode() {
		return mode_;
	}

	void set_mode(ProcessUnit mode) {
		mode_ = mode;
	}

#ifndef CPU_ONLY
	cublasHandle_t& CublasHandle() {
		return cublas_handle_;
	}

	curandGenerator_t& CurandGenerator() {
		return curand_generator_;
	}
#endif

private:
#ifndef CPU_ONLY
	Config(): cublas_handle_(nullptr), curand_generator_(nullptr) {
		Init();
	};
#else
	Config() {
		Init();
	};
#endif

	~Config() {
#ifndef CPU_ONLY
		if (cublas_handle_) cublasDestroy(cublas_handle_);
		if (curand_generator_) curandDestroyGenerator(curand_generator_);
#endif
	}

	void Init() {
		if (!init) {
#ifndef CPU_ONLY
			//init the cuda
			int count;
			cudaGetDeviceCount(&count);
			CHECK_GT(count, 0);
			//TODO:now, we only the one gpu
			cudaSetDevice(0);
			//init the cublas
			cublasCreate(&cublas_handle_);
			curandCreateGenerator(&curand_generator_, CURAND_RNG_PSEUDO_DEFAULT);
#endif
			init = true;
		}

	}

private:
	bool init = false;
	ProcessUnit mode_ = ProcessUnit::CPU;
#ifndef CPU_ONLY
	cublasHandle_t cublas_handle_;
	curandGenerator_t curand_generator_;
#endif
};
}




#endif
