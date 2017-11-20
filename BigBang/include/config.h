#ifndef CONFIG_H_
#define CONFIG_H_

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "base.h"
#include "gtest.h"

#ifndef CPU_ONLY



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
#endif

	private:
		Config() {
			Init();
		};

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
#endif
				init = true;
			}

		}

	private:
		bool init = false;
		cublasHandle_t cublas_handle_;
		ProcessUnit mode_ = ProcessUnit::GPU;
	};
}

#endif





#endif
