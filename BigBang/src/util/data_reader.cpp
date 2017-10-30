#include "../../include/util/data_reader.h"

#include "../../include/base.h"
#include "../../include/util/load_data.h"

namespace BigBang {

template<typename dtype>
void DataReader<dtype>::Read() {
	swtich(DataParams::DataSet) {
		case DataParams::DataSet::Mnist:
			ReadMnistImage<dtype>(params_.train_data__image_dir_, params_.begin_train_data_index_, params_.end_train_data_index_,
				params_.channels, params_.h_.params_.w_, train_data_image_.get());
			ReadMnistLabel<dtype>(params_.train_data_label_dir_, params_.begin_train_data_index_, params_.end_train_data_index_,
				params_.channels, params_.h_.params_.w_, train_data_image_.get());
			break;
		default:
			THROW_EXCEPTION;
	}
}

template<typename dtype>
void DataReader<dtype>::Load() {

}

}