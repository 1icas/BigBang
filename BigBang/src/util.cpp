#include "../include/util.h"

#include <iomanip>
#include <sstream>

namespace BigBang {
std::string convert_int_string(const int n, const int align) {
	std::ostringstream s;
	s << std::setw(align) << std::setfill('0') << n;
	return s.str();
}
}