#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

#include "base.h"
#include "log.h"
#include "synced_memory.h"

namespace BigBang {
// it's enough to use a 4 dimension tensor now(cnn and dnn)
// but we still use a vector to stroe the dimension info
template <typename dtype>
class Tensor {
public:
	DISABLE_COPY_AND_ASSIGNMENT(Tensor)

	explicit Tensor(const std::vector<int>& shape) :
		shape_(shape), dimension_(shape_.size()), size_(0) {
		if (dimension_) {
			size_ = shape_[0];
			for (int i = 1; i < dimension_; ++i) {
				size_ *= shape_[i];
			}
		}
		int byte_size = sizeof(dtype) * size_;
		data_ = std::make_shared<SyncedMemory>(byte_size);
		diff_data_ = std::make_shared<SyncedMemory>(byte_size);
	}

	int size() const {
		return size_;
	}

	int dimension() const {
		return dimension_;
	}

	int shape(int index) const{
		CHECK_GTE(index, 0);
		CHECK_LT(index, dimension_);
		return shape_[index];
	}

	const dtype* cpu_data() const {
		return reinterpret_cast<const dtype*>(data_->cpu_data());
	}

	const dtype* gpu_data() const {
		return reinterpret_cast<const dtype*>(data_->gpu_data());
	}

	const dtype* cpu_diff_data() const {
		return reinterpret_cast<const dtype*>(diff_data_->cpu_data());
	}

	const dtype* gpu_diff_data() const {
		return reinterpret_cast<const dtype*>(diff_data_->gpu_data());
	}

	dtype* mutable_cpu_data() {
		return static_cast<dtype*>(data_->mutable_cpu_data());
	}

	dtype* mutable_gpu_data() {
		return static_cast<dtype*>(data_->mutable_gpu_data());
	}

	dtype* mutable_cpu_diff_data() {
		return static_cast<dtype*>(diff_data_->mutable_cpu_data());
	}

	dtype* mutable_gpu_diff_data() {
		return static_cast<dtype*>(diff_data_->mutable_gpu_data());
	}

private:
	std::vector<int> shape_;
	int dimension_;
	int size_;

	std::shared_ptr<SyncedMemory> data_;
	std::shared_ptr<SyncedMemory> diff_data_;
};
}

#endif
