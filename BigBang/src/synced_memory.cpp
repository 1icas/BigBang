#include "../include/synced_memory.h"

#include "../include/base.h"
namespace BigBang {

	const void* SyncedMemory::cpu_data() {
		Check();
		GpuToCpu();
		return cpu_data_;
	}

	const void* SyncedMemory::gpu_data() {
		Check();
		CpuToGpu();
		return gpu_data_;
	}

	void* SyncedMemory::mutable_cpu_data() {
		void* data = const_cast<void*>(cpu_data());
		mem_state_ = SyncedState::AT_CPU;
		return data;
	}

	void* SyncedMemory::mutable_gpu_data() {
		void* data = const_cast<void*>(gpu_data());
		mem_state_ = SyncedState::AT_GPU;
		return data;
	}

	void SyncedMemory::CheckDevice() {
#ifndef CPU_ONLY
		int device = 0;
		cudaGetDevice(&device);
		CHECK_EQ(device_, device);
#endif
	}

	void SyncedMemory::CpuToGpu() {
#ifndef CPU_ONLY
		if (mem_state_ == SyncedState::UNINITIALIZED) {
			cudaMalloc(&gpu_data_, size_);
			cudaMemset(gpu_data_, 0, size_);
			mem_state_ = SyncedState::AT_GPU;
		}
		else if (mem_state_ == SyncedState::AT_CPU) {
			if (gpu_data_ == nullptr) {
				cudaMalloc(&gpu_data_, size_);
			}
			cudaMemcpy(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice);
			mem_state_ = SyncedState::SYNCED;
		}
#else
		NO_GPU
#endif
	}

	void SyncedMemory::GpuToCpu() {
#ifndef CPU_ONLY
		if (mem_state_ == SyncedState::UNINITIALIZED) {
			cpu_data_ = malloc(size_);
			bigbangcpumemset(cpu_data_, 0, size_);
			mem_state_ = SyncedState::AT_CPU;
		}
		else if (mem_state_ == SyncedState::AT_GPU) {
			if (cpu_data_ == nullptr) {
				cpu_data_ = malloc(size_);
			}
			cudaMemcpy(cpu_data_, gpu_data_, size_, cudaMemcpyDeviceToHost);
			mem_state_ = SyncedState::SYNCED;
		}
#endif
	}

#ifndef CPU_ONLY
	void SyncedMemory::async_gpu_data_push(const cudaStream_t& stream) {
		CHECK_EQ(mem_state_, SyncedState::AT_CPU);
		if (gpu_data_ == nullptr) {
			cudaMalloc(&gpu_data_, size_);
		}
		cudaMemcpyAsync(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice, stream);
		mem_state_ = SyncedState::SYNCED;
	}

#endif

	void SyncedMemory::Reset() {
		if (cpu_data_) {
			memset(cpu_data_, 0, size_);
		}
		if (gpu_data_) {
			cudaMemset(gpu_data_, 0, size_);
		}
	}



}