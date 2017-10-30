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
	
	Tensor() : shape_(), dimension_(0), 
		size_(0), data_(nullptr), diff_data_(nullptr){}

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
		check();
		CHECK_GTE(index, 0);
		CHECK_LT(index, dimension_);
		return shape_[index];
	}

	void set_shape(const std::vector<int>& shape) {
		if (!uninitialized()) return;
		shape_ = shape;
		dimension_ = shape_.size();
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
		
	const dtype* cpu_data() const {
		check();
		return reinterpret_cast<const dtype*>(data_->cpu_data());
	}

	const dtype* gpu_data() const {
		check();
		return reinterpret_cast<const dtype*>(data_->gpu_data());
	}

	const dtype* cpu_diff_data() const {
		check();
		return reinterpret_cast<const dtype*>(diff_data_->cpu_data());
	}

	const dtype* gpu_diff_data() const {
		check();
		return reinterpret_cast<const dtype*>(diff_data_->gpu_data());
	}

	dtype* mutable_cpu_data() {
		check();
		return static_cast<dtype*>(data_->mutable_cpu_data());
	}

	dtype* mutable_gpu_data() {
		check();
		return static_cast<dtype*>(data_->mutable_gpu_data());
	}

	dtype* mutable_cpu_diff_data() {
		check();
		return static_cast<dtype*>(diff_data_->mutable_cpu_data());
	}

	dtype* mutable_gpu_diff_data() {
		check();
		return static_cast<dtype*>(diff_data_->mutable_gpu_data());
	}
	
	bool uninitialized() const {
		return data_ == nullptr && diff_data_ == nullptr;
	}
private:
	void check() const {
		if (uninitialized()) {
			THROW_EXCEPTION;
		}
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
