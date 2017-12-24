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

void ParseBinaryFileToMessage(const std::string& file_name, Message* message) {
	int fd = open(file_name.c_str(), O_RDONLY | O_BINARY);
	CHECK_NE(fd, -1);
	ZeroCopyInputStream* input_stream = new FileInputStream(fd);
	CodedInputStream* coded_input_stream = new CodedInputStream(input_stream);
	coded_input_stream->SetTotalBytesLimit(INT_MAX, 536870912);
	bool success = message->ParseFromCodedStream(coded_input_stream);
	delete coded_input_stream;
	delete input_stream;
	close(fd);
	CHECK_EQ(success, true);
}

void ParseMessageToBinaryFile(const std::string& file_name, const Message& message) {
	int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0644);
	CHECK_NE(fd, -1);
	ZeroCopyOutputStream* output = new FileOutputStream(fd);
	CodedOutputStream* coded_output =	new CodedOutputStream(output);
	bool success = message.SerializeToCodedStream(coded_output);
	delete coded_output;
	delete output;
	close(fd);
	CHECK_EQ(success, true);
}




}