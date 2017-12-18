#include <cassert>
#include <fstream>
#include <memory>
#include <string>
#include <utility>


#include "../../proto/bigbang.pb.h"
#include "../../include/util.h"
#include "../../include/util/db_lmdb.h"

using namespace BigBang;
using namespace std;


uint32_t swap_endian(uint32_t val) {
	val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
	return (val << 16) | (val >> 16);
}


void convert_dataset(const char* image_filename, const char* label_filename,
	const char* db_path) {
	// Open files
	std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
	std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
	
	// Read the magic and the meta data
	uint32_t magic;
	uint32_t num_items;
	uint32_t num_labels;
	uint32_t rows;
	uint32_t cols;

	image_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2051);
	label_file.read(reinterpret_cast<char*>(&magic), 4);
	magic = swap_endian(magic);
	CHECK_EQ(magic, 2049);
	image_file.read(reinterpret_cast<char*>(&num_items), 4);
	num_items = swap_endian(num_items);
	label_file.read(reinterpret_cast<char*>(&num_labels), 4);
	num_labels = swap_endian(num_labels);
	CHECK_EQ(num_items, num_labels);
	image_file.read(reinterpret_cast<char*>(&rows), 4);
	rows = swap_endian(rows);
	image_file.read(reinterpret_cast<char*>(&cols), 4);
	cols = swap_endian(cols);

	std::shared_ptr<DB> db(new LMDB());
	db->Open(db_path, DBMode::CREATE);
	std::shared_ptr<Transaction> txn(db->CreateTransaction());

	// Storing to db
	char label;
	char* pixels = new char[rows * cols];
	int count = 0;
	string value;

	Datum datum;
	datum.set_channels(1);
	datum.set_height(rows);
	datum.set_width(cols);
	
	for (int item_id = 0; item_id < num_items; ++item_id) {
		image_file.read((char*)pixels, rows * cols);
		label_file.read(&label, 1);
		/*for (int k = 0; k < 784; ++k) {
			std::cout << (double)(pixels[k]) << std::endl;
		}*/
	
		datum.set_data(pixels, rows * cols);
		datum.set_label(label);
		string key_str = convert_int_string(item_id, 8);
		datum.SerializeToString(&value);
		txn->Put(key_str, value);

		if (++count % 1000 == 0) {
			txn->Commit();
		}
	}
	// write the last batch
	if (count % 1000 != 0) {
		txn->Commit();
	}
	
	delete[] pixels;
	db->Close();
}

//int main() {
//	//test
//	convert_dataset("D:/git/BigBang/BigBang/src/test/mnist_data/t10k-images.idx3-ubyte", 
//		"D:/git/BigBang/BigBang/src/test/mnist_data/t10k-labels.idx1-ubyte",
//		"D:/deeplearning/mnist_lmdb/mnist_test.mdb");
//	//train
//	convert_dataset("D:/git/BigBang/BigBang/src/test/mnist_data/train-images.idx3-ubyte",
//		"D:/git/BigBang/BigBang/src/test/mnist_data/train-labels.idx1-ubyte",
//		"D:/deeplearning/mnist_lmdb/mnist_train.mdb");
//}
