#ifndef DB_H
#define DB_H

#include <string>
#include "../../include/base.h"

namespace BigBang {

enum class DBMode {
	READ,
	WRITE,
	CREATE
};

class Cursor {
public:
	Cursor(){}
	virtual ~Cursor(){}
	virtual void SeekToFirst() = 0;
	virtual void Next() = 0;
	virtual std::string key() = 0;
	virtual std::string value() = 0;
	virtual bool valid() = 0;
	DISABLE_COPY_AND_ASSIGNMENT(Cursor);
};

class Transaction {
public:
	Transaction(){}
	virtual ~Transaction(){}
	virtual void Put(const std::string& key, const std::string& value) = 0;
	virtual void Commit() = 0;
	DISABLE_COPY_AND_ASSIGNMENT(Transaction);
};

class DB {
public:
	DB(){}
	virtual ~DB(){}
	virtual void Open(const std::string& dir, const DBMode& mode) = 0;	
	virtual void Close() = 0;
	virtual Cursor* CreateCursor() = 0;
	virtual Transaction* CreateTransaction() = 0;
	DISABLE_COPY_AND_ASSIGNMENT(DB);
};

}


#endif
