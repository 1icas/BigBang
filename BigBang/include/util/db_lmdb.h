#ifndef DB_LMDB_H
#define DB_LMDB_H

#include <vector>

#include "db.h"
#include "lmdb++.h"
#include "../gtest.h"

namespace BigBang {

class LMDBCursor : public Cursor {
public:
	LMDBCursor(MDB_txn* txn, MDB_cursor* cursor)
		: mdb_txn_(txn), mdb_cursor_(cursor) {
		SeekToFirst();
	}
	virtual ~LMDBCursor() {
		lmdb::cursor_close(mdb_cursor_);
		lmdb::txn_abort(mdb_txn_);
	}
	virtual void SeekToFirst() { Seek(MDB_FIRST); };
	virtual void Next() { Seek(MDB_NEXT); };
	virtual std::string key() {
		return std::string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
	};
	virtual std::string value() {
		return std::string(static_cast<const char*>(mdb_value_.mv_data),
			mdb_value_.mv_size);
	}
	virtual bool valid() {
		return valid_;
	}

private:
	void Seek(MDB_cursor_op op) {
		bool state = lmdb::cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
		if (state == false) {
			valid_ = false;
		}
		else {
			valid_ = true;
		}
	}

private:
	MDB_txn* mdb_txn_;
	MDB_cursor* mdb_cursor_;
	MDB_val mdb_key_, mdb_value_;
	bool valid_ = false;
};

class LMDBTransaction : public Transaction{
public:
	LMDBTransaction(MDB_env* mdb_env)
		: mdb_env_(mdb_env) {}
	virtual void Put(const std::string& key, const std::string& value);
	virtual void Commit();

private:
	void Reserve();

private:
	MDB_env* mdb_env_;
	std::vector<std::string> keys_, values_;
};

class LMDB : public DB {
public:
	LMDB(): mdb_env_(nullptr){}
	virtual ~LMDB() {
		Close();
	}
	virtual void Open(const std::string& dir, const DBMode& mode);

	virtual void Close() {
		if (mdb_env_) {
			lmdb::dbi_close(mdb_env_, mdb_dbi_);
			lmdb::env_close(mdb_env_);
			mdb_env_ = nullptr;
		}
	};
	virtual LMDBCursor* CreateCursor();
	virtual LMDBTransaction* CreateTransaction();

private:
	/*lmdb::env env_;
	MDB_env*/
	MDB_env* mdb_env_;
	MDB_dbi mdb_dbi_;
};

}







#endif

