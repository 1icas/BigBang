#ifndef THREAD_H
#define THREAD_H

#include <memory>
#include <thread>

class Thread {
public:
	Thread(): thread_(nullptr) {}
	virtual ~Thread() {
		Stop();
	}
	void Start();
	void Stop();
	bool Should_Stop();

protected:
	virtual void entry() {}

private:
	std::shared_ptr<std::thread> thread_;
};



#endif
