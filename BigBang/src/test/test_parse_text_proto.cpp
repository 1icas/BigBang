#include "test.h"

void Test::TestParseTextFileToProtobuf() {
	SolverParameter sp;
	sp.set_train_iterations(100);
	sp.set_lr(0.5);
	ParseMessageToTextFile("./src/test/solver_params.txt", sp);
	int end = 0;

	/*SolverParameter sp;
	ParseTextFileToMessage("./src/test/solver_params.txt", &sp);
	int iter = sp.train_iterations();
	double lr = sp.lr();*/
}