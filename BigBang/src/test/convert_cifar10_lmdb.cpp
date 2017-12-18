#include <cassert>
#include <fstream>
#include <memory>
#include <string>


#include "../../proto/bigbang.pb.h"
#include "../../include/util.h"
#include "../../include/util/db_lmdb.h"

using namespace BigBang;

const int kBatchNums = 5;
const int kBatchSize = 10000;
const int kPixels = 3 * 32 * 32;
const int height = 32;
const int width = 32;

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
}

void convert_cifar_compute_mean(const std::string& input_folder, const std::string& output_folder) {
	std::shared_ptr<DB> lmdb(new LMDB());
	lmdb->Open(output_folder + "cifar10.mdb",
		DBMode::CREATE);
	std::shared_ptr<Transaction> transaction(lmdb->CreateTransaction());

	char data[kPixels];
	Datum datum;
	datum.set_channels(3);
	datum.set_height(height);
	datum.set_width(width);
	const int t = kBatchNums + 1;
	for (int i = 0; i < t; ++i) {
		std::string file_name = input_folder + (i < kBatchNums ? ("/data_batch_" + convert_int_string(i + 1)) : "/test_batch")
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
}

void compute_cifar_mean(const std::string& file, const std::string& output) {
	std::shared_ptr<DB> lmdb(new LMDB());
	lmdb->Open(file, DBMode::READ);
	std::shared_ptr<Cursor> cursor(lmdb->CreateCursor());
	Datum datum;
	datum.ParseFromString(cursor->value());
	
	TensorProto tp;
	auto shape = tp.mutable_shape();
	shape->add_dim(1);
	shape->add_dim(datum.channels());
	shape->add_dim(datum.height());
	shape->add_dim(datum.width());
	const int size = 1 * datum.channels() * datum.height() * datum.width();
	for (int i = 0; i < size; ++i) {
		tp.add_f_data(0.);
	}
	int count = 0;
	while (cursor->valid()) {
		Datum datum;
		datum.ParseFromString(cursor->value());
		const std::string& data = datum.data();

		if (!data.empty()) {
			for (int i = 0; i < size; ++i) {
				tp.set_f_data(i, tp.f_data(i) + data[i]);
			}
		}
		else {
			for (int i = 0; i < size; ++i) {
				tp.set_f_data(i, tp.f_data(i) + datum.f_data(i));
			}
		}
		++count;
		cursor->Next();
	}

	for (int i = 0; i < size; ++i) {
		tp.set_f_data(i, tp.f_data(i) / count);
	}

	std::fstream fs(output, std::ios::out | std::ios::trunc | std::ios::binary);
	tp.SerializeToOstream(&fs);
}



//int main() {
	//convert_cifar_compute_mean("D:/deeplearning/cifar-10-batches-bin", "D:/deeplearning/cifar_lmdb/");
	//convert("D:/deeplearning/cifar-10-batches-bin", "D:/deeplearning/cifar_lmdb");
	//test_read();
	//compute_cifar_mean("D:/deeplearning/cifar_lmdb/cifar10.mdb", "D:/deeplearning/cifar_lmdb/cifar10.mean");
//}