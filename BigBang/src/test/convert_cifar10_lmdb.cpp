#include <cassert>
#include <fstream>
#include <memory>
#include <iomanip>
#include <string>
#include <sstream>

#include "../../proto/bigbang.pb.h"
#include "../../include/util/db_lmdb.h"

using namespace BigBang;

const int kBatchNums = 5;
const int kBatchSize = 10000;
const int kPixels = 3 * 32 * 32;
const int height = 32;
const int width = 32;

std::string convert_int_string(const int n, const int align = 0) {
	std::ostringstream s;
	s << std::setw(align) << std::setfill('0') << n;
	return s.str();
}


void convert(const std::string& input_folder, const std::string& output_folder) {

	auto read_write = [&](bool is_train) {
		std::shared_ptr<DB> lmdb(new LMDB());
		lmdb->Open(output_folder + (is_train ? "/cifar10_train" : "/cifar10_test") + ".mdb", 
			DBMode::CREATE);
		std::shared_ptr<Transaction> transaction(lmdb->CreateTransaction());

		char data[kPixels];
		Datum datum;
		datum.set_channels(3);
		datum.set_height(height);
		datum.set_width(width);
		const int t = is_train ? kBatchNums : 1;
		for (int i = 0; i < t; ++i) {
			std::string file_name = input_folder + (is_train ? ("/data_batch_" + convert_int_string(i+1)) : "/test_batch")
				+ ".bin";
			std::ifstream fstream(file_name.c_str(), std::ios::in | std::ios::binary);
			if (!fstream) assert(false);
			char label = 0;
			for (int k = 0; k < kBatchSize; ++k) {
				fstream.read(&label, 1);
				fstream.read(data, kPixels);
				datum.set_label(label);
				datum.set_data(data, kPixels);
				std::string out;
				datum.SerializeToString(&out);
				transaction->Put(convert_int_string(i * kBatchNums + k, 5), out);
			}
		}
		transaction->Commit();
		lmdb->Close();
	};

	read_write(true);
	read_write(false);
}

void test_read() {
	std::shared_ptr<DB> lmdb(new LMDB());
	lmdb->Open("D:/deeplearning/cifar_lmdb/cifar10_train.mdb", DBMode::READ);
	std::shared_ptr<Cursor> cursor(lmdb->CreateCursor());
	std::string key = cursor->key();
	std::string value = cursor->value();
	int end = 0;
}

//int main() {
//	convert("D:/deeplearning/cifar-10-batches-bin", "D:/deeplearning/cifar_lmdb");
//	test_read();
//}