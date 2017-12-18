#include "test.h"

template<typename dtype>
void Test::TestCifar10Model() {
	SolverParameter sp;
	ParseTextFileToMessage("./src/test/train_cifar10_solver.prototxt", &sp);
	Solver<dtype> solver(sp);
	solver.Train();
}
template void Test::TestCifar10Model<float>();
template void Test::TestCifar10Model<double>();
