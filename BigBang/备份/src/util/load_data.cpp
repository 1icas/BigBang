#include "../../include/util/load_data.h"

#include <fstream>
using namespace std;

inline int ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

namespace BigBang {

template<typename dtype>
void ReadMnistImage(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<dtype>* m) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		file.read((char*)&n_rows, sizeof(n_rows));
		file.read((char*)&n_cols, sizeof(n_cols));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		n_rows = ReverseInt(n_rows);
		n_cols = ReverseInt(n_cols);
		dtype* data = m->mutable_cpu_data();
		const int pixels_num = n_rows * n_cols;
		file.seekg(start_data_index*n_rows*n_cols+16, ios::beg);
		const int n = end_data_index - start_data_index;
		for (int i = 0; i < n; i++) {
			for (int r = 0; r < n_rows; r++) {
				for (int c = 0; c < n_cols; c++) {
					unsigned char image = 0;
					file.read((char*)&image, sizeof(image));
					data[i*pixels_num + n_cols*r + c] = (dtype)(image) / (dtype)255.;
				}
			}

		}
	}
	file.close();
}

template void ReadMnistImage<float>(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<float>* m);
template void ReadMnistImage<double>(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<double>* m);

template<typename dtype>
void ReadMnistLabel(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<dtype>* m) {
	std::ifstream file(filename, std::ios::binary);
	if (file.is_open()) {
		int magic_number = 0;
		int number_of_images = 0;
		file.read((char*)&magic_number, sizeof(magic_number));
		file.read((char*)&number_of_images, sizeof(number_of_images));
		magic_number = ReverseInt(magic_number);
		number_of_images = ReverseInt(number_of_images);
		dtype* data = m->mutable_cpu_data();
		file.seekg(start_data_index+8, ios::beg);
		const int n = end_data_index - start_data_index;
		for (int i = 0; i < n; i++) {
			unsigned char label = 0;
			file.read((char*)&label, sizeof(label));
			data[i * 10 + label] = static_cast<dtype>(1.0);
		}
	}
	file.close();
}

template void ReadMnistLabel<float>(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<float>* m);
template void ReadMnistLabel<double>(const std::string& filename, const int start_data_index, const int end_data_index,
	const int channels, const int h, const int w, Tensor<double>* m);

}