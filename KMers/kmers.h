#pragma once

#include <algorithm>
#include <iostream>

using namespace std;

struct PosCount {
	int pos;
	int count;

	PosCount(int p, int c) : pos(p), count(c) {}

	PosCount() : pos(0), count(0) {}

	bool operator == (const PosCount& a) const {
		return pos == a.pos && count == a.count;
	}

	bool operator != (const PosCount& a) const {
		return pos != a.pos || count != a.count;
	}

	bool operator < (const PosCount& a) const {
		if (pos != a.pos) return pos < a.pos;
		return count < a.count;
	}
};

class Result {
public:
	Result(int num, PosCount* p) : n(num), pos_count(p) {}

	Result(int num) : n(num) {
		this->pos_count = new PosCount[n];
	}

	~Result() {
		delete pos_count;
	}

	int get_position(int i) {
		if(i < 0 || i >= n)
			return -1;
		return pos_count[i].pos;
	}

	int get_count(int i) {
		if(i < 0 || i >= n)
			return -1;
		return pos_count[i].count;
	}

	bool set_position(int i, int v) {
		if(i < 0 || i >= n)
			return false;
		pos_count[i].pos = v;
		return true;
	}

	int set_count(int i, int v) {
		if(i < 0 || i >= n)
			return false;
		pos_count[i].count = v;
		return true;
	}

	void sort() {
		std::sort(pos_count, pos_count + n);
	}

	bool equal(const Result& res) {
		if(n != res.n) return false;
		for(int i=0; i<n; ++i)
			if(pos_count[i] != res.pos_count[i])
				return false;
		return true;
	}

	void print() const {
		cout << "==================" << endl;
		for(int i=0; i<n; ++i) {
			cout << pos_count[i].pos << "\t" << pos_count[i].count << endl;
		}
		cout << "==================" << endl;
	}

private:
	int n;
	PosCount* pos_count;
};

class Data {
public:
	Data(int num, char* input) : n(num), data(input) {}

	~Data() {
		delete data;
	}

	const char* get_data() const {
		return data;
	}

	int get_size() const {
		return n;
	}

private:
	int n;
	char* data;
};
