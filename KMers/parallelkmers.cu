
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/copy.h>
#include <iostream>

#include "parallelkmers.cuh"

template <typename KeyVector, typename PermutationVector>
void update_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // temporary storage for keys
    KeyVector temp(keys.size());

    // permute the keys with the current reordering
    thrust::gather(permutation.begin(), permutation.end(), keys.begin(), temp.begin());

    // stable_sort the permuted keys and update the permutation
    thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation.begin());
}


template <typename KeyVector, typename PermutationVector>
void apply_permutation(KeyVector& keys, PermutationVector& permutation)
{
    // copy keys to temporary vector
    KeyVector temp(keys.begin(), keys.end());

    // permute the keys
    thrust::gather(permutation.begin(), permutation.end(), temp.begin(), keys.begin());
}

struct cmp_data_functor
{
	cmp_data_functor(int* pos,
		char* data) : pos(pos), data(data) {}

	const int* pos;
	const char* data;

    __host__ __device__
    void operator()(int& i)
    {
		if(i == 0) i = 0;
		else {
			if (data[pos[i-1]] == data[pos[i]]) i = 0;
			else i = 1;
		}
    }
};

struct cmp_rk_functor
{
	cmp_rk_functor(int* pos,
		int* rk, int len) : pos(pos), rk(rk), gap(0), len(len) {}

	const int* pos;
	const int* rk;
	int gap;
	const int len;

    __host__ __device__
    void operator()(int& i)
    {
		if(i == 0) i = 0;
		else {
			int r1 = pos[i-1] + gap >= len ? len : rk[pos[i-1] + gap];
			int r2 = pos[i] + gap >= len ? len : rk[pos[i] + gap];

			if(rk[pos[i-1]] == rk[pos[i]] && r1 == r2) i = 0;
			else i = 1;
		}
    }
};

struct second_half_functor
{
	second_half_functor(int* pos,
		int* rk, int len) : pos(pos), rk(rk), gap(0), len(len) {}

	const int* pos;
	const int* rk;
	int gap;
	const int len;

    __host__ __device__
    void operator()(int& i)
    {
		if(i + gap >= len) i = -1;
		else i = rk[i + gap];
    }
};

struct first_half_functor
{
	first_half_functor(int* pos,
		int* rk) : pos(pos), rk(rk) {}

	const int* pos;
	const int* rk;

    __host__ __device__
    void operator()(int& i)
    {
		i = rk[i];
    }
};

Result parallel_kmers(const Data& input, int k) {
	const int len = input.get_size();
	const char * data = input.get_data();

	// Copy data
	thrust::host_vector<int> host_data(data, data+len);
	thrust::device_vector<char> dev_data(len);
	thrust::copy(host_data.begin(), host_data.end() , dev_data.begin());

	// Initialize pos & rk
	thrust::device_vector<int> pos(len);
	thrust::device_vector<int> rk(len);
	
	thrust::sequence(pos.begin(), pos.end());

	update_permutation(dev_data, pos);

	thrust::device_vector<int> id(len);
	thrust::sequence(id.begin(), id.end());
	thrust::for_each(id.begin(), id.end(), cmp_data_functor(pos.data().get(), dev_data.data().get()));
	thrust::inclusive_scan(id.begin(), id.end(), id.begin());
	thrust::scatter(id.begin(), id.end(), pos.begin(), rk.begin());

	second_half_functor cmp1(pos.data().get(), rk.data().get(), len);
	first_half_functor cmp2(pos.data().get(), rk.data().get());
	cmp_rk_functor func(pos.data().get(), rk.data().get(), len);

	for(int gap = 1; gap < k; gap <<= 1) {
		// Sort pos using rk[pos[i] + k]
		thrust::sequence(id.begin(), id.end());
		cmp1.gap = gap;
		thrust::for_each(id.begin(), id.end(), cmp1);
		update_permutation(id, pos);

		// Sort pos using rk[pos[i]]
		thrust::sequence(id.begin(), id.end());
		thrust::for_each(id.begin(), id.end(), cmp2);
		update_permutation(id, pos);

		// Update rk
		thrust::sequence(id.begin(), id.end());
		func.gap = gap;
		thrust::for_each(id.begin(), id.end(), func);

		thrust::inclusive_scan(id.begin(), id.end(), id.begin());
		thrust::scatter(id.begin(), id.end(), pos.begin(), rk.begin());

	}

	thrust::host_vector<int> host_pos = pos;
	thrust::host_vector<int> host_rk = rk;
	const int * p_pos = host_pos.data();
	const int * p_rk = host_rk.data();

	PosCount* res = new PosCount[len];
	int cnt = 1;
	int j = 0;
	res[0].pos = pos[0];
	for(int i=1; i<len; ++i) {
		if(p_rk[p_pos[i-1]] == p_rk[p_pos[i]]) {
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

	return Result(j, res);
}
