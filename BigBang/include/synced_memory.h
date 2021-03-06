#ifndef SYNCED_MEMORY_H
#define SYNCED_MEMORY_H

#include <memory>
#include <stdlib.h>

#include <cuda_runtime.h>

#include "base.h"
#include "gtest.h"
//#include "log.h"
#include "util/common.h"

namespace BigBang {

class SyncedMemory {
public:
	explicit SyncedMemory(size_t size) :
		size_(size), cpu_data_(nullptr), 
		gpu_data_(nullptr), mem_state_(UNINITIALIZED) {
		CHECK_LT(size, MAX_CAPACITY);
#ifndef CPU_ONLY
		CUDA_CHECK(cudaGetDevice(&device_));
#endif
	}
	~SyncedMemory() {
		if (cpu_data_) free(cpu_data_);
		if (gpu_data_) cudaFree(gpu_data_);
	}

	enum SyncedState { UNINITIALIZED, AT_GPU, AT_CPU, SYNCED };

	const void* cpu_data();
	const void* gpu_data();
	void* mutable_cpu_data();
	void* mutable_gpu_data();
	void Reset();

#ifndef CPU_ONLY
	void async_gpu_data_push(const cudaStream_t& stream);
#endif

private:
	void Check() {
		CheckDevice();
	}

	/*void Init() {
		if (mem_state_ == SyncedState::UNINITIALIZED) {
			cpu_data_ = malloc(size_);
			bigbangcpumemset(cpu_data_, 0, size_);
#ifndef CPU_ONLY
			cudaMalloc(&gpu_data_, size_);
			cudaMemset(gpu_data_, 0, size_);
#endif
			mem_state_ = SyncedState::SYNCED;
		}
	}*/

	void CheckDevice();
	void CpuToGpu();
	void GpuToCpu();
	
private:
	size_t size_;
	void* cpu_data_;
	void* gpu_data_;
	SyncedState mem_state_;
	int device_;
};

}






#endif