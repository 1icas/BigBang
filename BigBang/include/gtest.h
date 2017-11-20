#ifndef GTEST_H
#define GTEST_H

#define CHECK_OP(op, val1, val2) \
	if(!(val1 op val2)) assert(false);

#define CHECK_EQ(a, b) \
	CHECK_OP(==, a, b)

#define CHECK_GT(a, b) \
	CHECK_OP(>, a, b)

#define CHECK_LT(a, b) \
	CHECK_OP(<, a, b)

#define CHECK_GTE(a, b) \
	CHECK_OP(>=, a, b)

#define CHECK_LTE(a, b) \
	CHECK_OP(<=, a, b)

#define THROW_EXCEPTION assert(false)

#define VALIDATE_POINTER(p) \
	assert(p != nullptr);


#endif
