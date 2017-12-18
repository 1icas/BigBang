#include "test.h"

template<typename dtype>
void Test::TestConvLayerNetwork() {
	SolverParameter sp;
	ParseTextFileToMessage("./src/test/train_mnist_lenet_solver.prototxt", &sp);
	Solver<dtype> solver(sp);
	solver.Train();
}
template void Test::TestConvLayerNetwork<float>();
template void Test::TestConvLayerNetwork<double>();
