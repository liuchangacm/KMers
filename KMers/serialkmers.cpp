#include "serialkmers.h"
#include <algorithm>

const char * glob_data;
int K;

Result serial_kmers(const Data& input, int k) {
	const int len = input.get_size();
	const char * data = input.get_data();

	int* pos = new int[len];
	int* rk = new int[len];
	for(int i=0; i<len; ++i)
		pos[i] = i;

	// Initialize the first phase using a counting sort
	int* count = new int[len+1 > 26 ? len + 1 : 26];
	bool* ident = new bool[len];

	int* new_pos = new int[len];

	for(int i=0; i<26; ++i) count[i] = 0;
	for(int i=0; i<len; ++i)
		count[data[i] - 'A'] ++;
	for(int i=1; i<26; ++i) {
		count[i] += count[i-1];
	}
	for(int i=len-1; i>=0; --i) {
		new_pos[--count[data[i] - 'A']] = i;
	}
	ident[0] = false;
	for(int i=1; i<len; ++i)
		ident[i] = data[new_pos[i]] == data[new_pos[i-1]];

	int* tmp_pos;

	tmp_pos = pos; pos = new_pos; new_pos = tmp_pos;
	int t_rk = 0;
	for(int i=0; i<len; ++i) {
		rk[pos[i]] = t_rk;
		if(i + 1 < len)
			t_rk += ident[i + 1] ? 0 : 1;
	}

	for(int gap = 1; gap < k; gap <<= 1) {
		// Sort pos using rk[pos[i] + k]
		memset(count, 0, sizeof(int) * (len + 1));
		for(int i=0; i<len; ++i) {
			if(pos[i] + gap >= len)
				count[0] ++;
			else
				count[rk[pos[i] + gap] + 1] ++;
		}
		for(int i=1; i<=len; ++i)
			count[i] += count[i-1];
		for(int i=len-1; i>=0; --i) {
			new_pos[--count[pos[i] + gap >= len ? 0 : rk[pos[i] + gap] + 1]] = pos[i];
		}
		tmp_pos = pos; pos = new_pos; new_pos = tmp_pos;
				
		//for(int i=0; i<len && i < 100; ++i) cout << "cpu " << pos[i] << "\t" << rk[i] << endl;

		// Sort pos using rk[pos[i]]
		memset(count, 0, sizeof(int) * (len + 1));
		for(int i=0; i<len; ++i) {
			count[rk[pos[i]]] ++;
		}
		for(int i=1; i<=len; ++i)
			count[i] += count[i-1];
		for(int i=len-1; i>=0; --i) {
			new_pos[--count[rk[pos[i]]]] = pos[i];
		}
		tmp_pos = pos; pos = new_pos; new_pos = tmp_pos;

		// Update ident
		ident[0] = false;
		for(int i=1; i<len; ++i) {
			int r1 = pos[i-1] + gap >= len ? len : rk[pos[i-1] + gap];
			int r2 = pos[i] + gap >= len ? len : rk[pos[i] + gap];

			ident[i] = rk[pos[i-1]] == rk[pos[i]] && r1 == r2;
		}

		// Update rk
		t_rk = 0;
		for(int i=0; i<len; ++i) {
			rk[pos[i]] = t_rk;
			if(i + 1 < len)
				t_rk += ident[i + 1] ? 0 : 1;
		}
	}
	
	PosCount* res = new PosCount[len];
	int cnt = 1;
	int j = 0;
	res[0].pos = pos[0];
	for(int i=1; i<len; ++i) {
		if(ident[i]) {
			++ cnt;
		} else {
			res[j].count = cnt;
			j ++;
			res[j].pos = pos[i];
			cnt = 1;
		}
	}
	res[j].count = cnt;
	j ++;

	delete pos;
	delete rk;
	delete count;
	delete ident;
	delete new_pos;

	return Result(j, res);
}

bool cmp(int pos1, int pos2) {
	for(int i=0; i<K; ++i)
		if(glob_data[pos1 + i] != glob_data[pos2 + i])
			return glob_data[pos1 + i] < glob_data[pos2 + i];
	return pos1 < pos2;
}

bool equal(int pos1, int pos2) {
	for(int i=0; i<K; ++i)
		if(glob_data[pos1 + i] != glob_data[pos2 + i])
			return false;
	return true;
}

Result naive_kmers(const Data& input, int k) {
	int len = input.get_size();
	glob_data = input.get_data();
	K = k;

	int* pos = new int[len];
	for(int i=0; i<len; ++i)
		pos[i] = i;
	std::sort(pos, pos + len, cmp);
	
	PosCount* res = new PosCount[len];
	int cnt = 1;
	int j = 0;
	res[0].pos = pos[0];
	for(int i=1; i<len; ++i) {
		if(equal(pos[i-1], pos[i])) {
			++ cnt;
		} else {
			res[j].count = cnt;
			j ++;
			res[j].pos = pos[i];
			cnt = 1;
		}
	}
	res[j].count = cnt;
	j ++;
	delete pos;
	return Result(j, res);
}

