#include "../../include/util/parse.h"

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include "../../include/util/unistd.h"
#include "../../include/gtest.h"

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace BigBang {

void ParseTextFileToMessage(const std::string& file_name, Message* message) {
	int fd = open(file_name.c_str(), O_RDONLY);
	CHECK_NE(fd, -1);
	FileInputStream* input = new FileInputStream(fd);
	bool success = google::protobuf::TextFormat::Parse(input, message);
	delete input;
	close(fd);
	CHECK_EQ(success, true);
}

void ParseMessageToTextFile(const std::string& file_name, const Message& message) {
	int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
	CHECK_NE(fd, -1);
	FileOutputStream* output = new FileOutputStream(fd);
	bool success = google::protobuf::TextFormat::Print(message, output);
	delete output;
	close(fd);
	CHECK_EQ(success, true);
}


}