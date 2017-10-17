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
		if (mem_state_ == SyncedState::AT_CPU) {
			cudaMemcpy(gpu_data_, cpu_data_, size_, cudaMemcpyHostToDevice);
			mem_state_ = SyncedState::SYNCED;
		}
#else
		NO_GPU
#endif
	}

	void SyncedMemory::GpuToCpu() {
#ifndef CPU_ONLY
		if (mem_state_ == SyncedState::AT_GPU) {
			cudaMemcpy(cpu_data_, gpu_data_, size_, cudaMemcpyDeviceToHost);
			mem_state_ = SyncedState::SYNCED;
		}
#endif
	}




}