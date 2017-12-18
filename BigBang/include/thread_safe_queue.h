#ifndef THREAR_SAFE_QUEUE_H
#define THREAR_SAFE_QUEUE_H

#include <mutex>
#include <queue>

//the inspriation from 
//https://www.justsoftwaresolutions.co.uk/threading/implementing-a-thread-safe-queue-using-condition-variables.html

template <typename dtype>
class ThreadSafeQueue {
public:

	void pop() {
		std::lock_guard<std::mutex> guard(mutex_);
		queue_.pop();
	}
	dtype front() {
		std::lock_guard<std::mutex> guard(mutex_);
		return queue_.front();
	}
	void push(const dtype& v) {
		{
			std::lock_guard<std::mutex> guard(mutex_);
			queue_.push(v);
		}
		cv_.notify_one();
	}
	void wait_and_pop(dtype& v) {
		std::unique_lock<std::mutex> ul(mutex_);
		while (queue_.empty()) {
			cv_.wait(ul);
		}
		v = queue_.front();
		queue_.pop();		
	}
	bool empty() {
		std::lock_guard<std::mutex> guard(mutex_);
		return queue_.empty();
	}
	/*bool try_pop() {

	}*/

private:
	std::queue<dtype> queue_;
	std::mutex mutex_;
	std::condition_variable cv_;
};









#endif
