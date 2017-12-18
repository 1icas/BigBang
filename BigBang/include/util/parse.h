#ifndef PARSE_H
#define PARSE_H

#include <string>

#include "../../proto/bigbang.pb.h"

using google::protobuf::Message;

namespace BigBang {

void ParseTextFileToMessage(const std::string& file_name, Message* message);
void ParseMessageToTextFile(const std::string& file_name, const Message& message);

}








#endif
