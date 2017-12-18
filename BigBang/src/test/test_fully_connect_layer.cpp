#include "test.h"

template<typename dtype>
void Test::TestFullyConnectLayer() {
	SolverParameter sp;
	ParseTextFileToMessage("./src/test/train_mnist_solver.prototxt", &sp);
	Solver<dtype> solver(sp);
	solver.Train();
}
template void Test::TestFullyConnectLayer<float>();
template void Test::TestFullyConnectLayer<double>();
