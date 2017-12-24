#ifndef TENSOR_H
#define TENSOR_H

#include <memory>
#include <vector>

#include "gtest.h"
#include "log.h"
#include "synced_memory.h"
#include "../proto/bigbang.pb.h"

namespace BigBang {
// it's enough to use a 4 dimension tensor now(cnn and dnn)
// but we still use a vector to stroe the dimension info
template <typename dtype>
class Tensor {
public:
	DISABLE_COPY_AND_ASSIGNMENT(Tensor)
	
	Tensor() : shape_(), dimension_(0), 
		size_(0), data_offset_(0), diff_data_offset_(0),
		data_(nullptr), diff_data_(nullptr){}

	explicit Tensor(const std::vector<int>& shape) :
		shape_(shape), dimension_(shape_.size()), size_(0), data_offset_(0),
		diff_data_offset_(0){
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

	inline int size() const {
		return size_;
	}

	inline int dimension() const {
		return dimension_;
	}

	inline int shape(int index) const{
		check();
		CHECK_GTE(index, 0);
		CHECK_LT(index, dimension_);
		return shape_[index];
	}

	inline std::vector<int> shape() const {
		return shape_;
	}

	void Reshape(const std::vector<int>& shape) {
		const int n = shape.size();
		if (!n) THROW_EXCEPTION;
		int new_size = shape[0];
		for (int i = 1; i < n; ++i) {
			new_size *= shape[i];
		}
		if (uninitialized() || size_ < new_size) {
			const int byte_size = new_size * sizeof(dtype);
			data_.reset(new SyncedMemory(byte_size));
			diff_data_.reset(new SyncedMemory(byte_size));
		}
		dimension_ = n;
		size_ = new_size;
		shape_ = shape;
	}
		
	const dtype* cpu_data() const {
		check();
		return (reinterpret_cast<const dtype*>(data_->cpu_data())) + data_offset_;
	}

	const dtype* gpu_data() const {
		check();
		return (reinterpret_cast<const dtype*>(data_->gpu_data())) + data_offset_;
	}

	const dtype* cpu_diff_data() const {
		check();
		return (reinterpret_cast<const dtype*>(diff_data_->cpu_data())) + diff_data_offset_;
	}

	const dtype* gpu_diff_data() const {
		check();
		return (reinterpret_cast<const dtype*>(diff_data_->gpu_data())) + diff_data_offset_;
	}

	dtype* mutable_cpu_data() {
		check();
		return (static_cast<dtype*>(data_->mutable_cpu_data())) + data_offset_;
	}

	dtype* mutable_gpu_data() {
		check();
		return (static_cast<dtype*>(data_->mutable_gpu_data())) + data_offset_;
	}

	dtype* mutable_cpu_diff_data() {
		check();
		return (static_cast<dtype*>(diff_data_->mutable_cpu_data())) + diff_data_offset_;
	}

	dtype* mutable_gpu_diff_data() {
		check();
		return (static_cast<dtype*>(diff_data_->mutable_gpu_data())) + diff_data_offset_;
	}
	
	bool uninitialized() const {
		return data_ == nullptr && diff_data_ == nullptr;
	}

	const std::shared_ptr<SyncedMemory> data() const {
		return data_;
	}

	const std::shared_ptr<SyncedMemory> diff_data() const {
		return diff_data_;
	}

	void shared_data(const Tensor<dtype>& other) {
		check();
		data_ = other.data();
	}

	void shared_diff_data(const Tensor<dtype>& other) {
		check();
		diff_data_ = other.diff_data();
	}

	void set_data_offset(int offset) {
		data_offset_ = offset;
	}

	void set_diff_data_offset(int offset) {
		diff_data_offset_ = offset;
	}

	void Reset() {
		if (data_) data_->Reset();
		if (diff_data_) diff_data_->Reset();
	}

	//now we only serialize the data_
	void Serialize(TensorProto* tp) {
		auto shape = tp->mutable_shape();
		for (int i = 0; i < shape_.size(); ++i) {
			shape->add_dim(shape_[i]);
		}
		const dtype* data = cpu_data();
		for (int i = 0; i < size_; ++i) {
			tp->add_d_data(data[i]);
		}
	}

	//we only handle the data_ now
	void Deserialize(const TensorProto* tp) {
		const int size = tp->d_data_size();
		dtype* cpu_mutable_data = mutable_cpu_data();
		for (int i = 0; i < size; ++i) {
			cpu_mutable_data[i] = tp->d_data(i);
		}
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
	int data_offset_;
	int diff_data_offset_;

	std::shared_ptr<SyncedMemory> data_;
	std::shared_ptr<SyncedMemory> diff_data_;
};
}

#endif
