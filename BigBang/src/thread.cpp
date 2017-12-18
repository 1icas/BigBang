#include "../include/thread.h"

void Thread::Start() {
	thread_.reset(new std::thread(&Thread::entry, this));
}

void Thread::Stop() {
	if (thread_ && thread_->joinable())
		thread_->join();
}

bool Thread::Should_Stop() {
	return thread_ && (!thread_->joinable());
}

