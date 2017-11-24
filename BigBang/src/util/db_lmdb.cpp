#include "../../include/util/db_lmdb.h"
#include <fstream>
#include "../../include/gtest.h"

bool check_file_exist(const std::string& dir) {
	std::ifstream f(dir);
	return f.good();
}

namespace BigBang {

void LMDBTransaction::Put(const std::string& key, const std::string& value) {
	keys_.push_back(key);
	values_.push_back(value);
}

void LMDBTransaction::Commit() {
	MDB_dbi mdb_dbi;
	MDB_val mdb_key, mdb_data;
	MDB_txn *mdb_txn;

	// Initialize MDB variables
	lmdb::txn_begin(mdb_env_, NULL, 0, &mdb_txn);
	lmdb::dbi_open(mdb_txn, NULL, 0, &mdb_dbi);

	for (int i = 0; i < keys_.size(); i++) {
		mdb_key.mv_size = keys_[i].size();
		mdb_key.mv_data = const_cast<char*>(keys_[i].data());
		mdb_data.mv_size = values_[i].size();
		mdb_data.mv_data = const_cast<char*>(values_[i].data());

		// Add data to the transaction
		bool state = lmdb::dbi_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);

		//now, we can just think this refer to (MDB_MAP_FULL equals to put_rc == false)
		if (state == false) {
			// Out of memory - double the map size and retry
			lmdb::txn_abort(mdb_txn);
			lmdb::dbi_close(mdb_env_, mdb_dbi);
			Reserve();
			Commit();
			return;
		}
	}

	// Commit the transaction
	lmdb::txn_commit(mdb_txn);

	// Cleanup after successful commit
	lmdb::dbi_close(mdb_env_, mdb_dbi);
	keys_.clear();
	values_.clear();
}

void LMDBTransaction::Reserve() {
	MDB_envinfo info;
	lmdb::env_info(mdb_env_, &info);
	lmdb::env_set_mapsize(mdb_env_, info.me_mapsize * 2);
}


void LMDB::Open(const std::string& dir, const DBMode& mode) {
	lmdb::env_create(&mdb_env_);
	if (mode == DBMode::CREATE && check_file_exist(dir)) {
		THROW_EXCEPTION;
	}
	int flag = 0;
	if (mode == DBMode::READ) {
		flag = MDB_RDONLY | MDB_NOTLS;
	}
	lmdb::env_open(mdb_env_, dir.c_str(), flag, 0664);
}

LMDBCursor* LMDB::CreateCursor() {
	MDB_txn* mdb_txn;
	MDB_cursor* mdb_cursor;
	lmdb::txn_begin(mdb_env_, nullptr, MDB_RDONLY, &mdb_txn);
	lmdb::dbi_open(mdb_txn, nullptr, 0, &mdb_dbi_);
	lmdb::cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor);
	return new LMDBCursor(mdb_txn, mdb_cursor);
}

LMDBTransaction* LMDB::CreateTransaction() {
	return new LMDBTransaction(mdb_env_);
}


}