#include "serialkmers.h"
#include "parallelkmers.cuh"
#include <random>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ctime>
#include <fstream>

using namespace std;

#define cout fout

Data * generateRandomInput(int n) {
	random_device dev;
	mt19937 rnd(dev());

	//mt19937 rnd;

	char* data = new char[n+1];
	for(int i=0; i<n; ++i)
		data[i] = rnd() % 4 + 'A';
	data[n] = 0;

	return new Data(n, data);
}

bool sanity_check(int size, int k) {
	Data* input = generateRandomInput(size);
	Result res1 = serial_kmers(*input, k);
	Result res2 = naive_kmers(*input, k);
	Result res3 = parallel_kmers(*input, k);
	bool ret = res1.equal(res2) && res1.equal(res3);
	delete input;
	return ret;
}

void test(int size, int k, ostream& out) {
	Data* input = generateRandomInput(size);

	clock_t start = clock();
	Result res1 = serial_kmers(*input, k);
	out << size << "\t" << k << "\tcpu time: " << (clock() - start) << endl;

	start = clock();
	Result res2 = parallel_kmers(*input, k);
	cudaDeviceSynchronize();
	out << size << "\t" << k << "\tgpu time: " << (clock() - start) << endl;

	delete input;
}

int main() {
	ofstream fout("result.txt");

	sanity_check(1000, 8);

	for(int times = 0; times < 10; ++ times) {
		for(int lvl = 15; lvl <= 25; ++lvl) {
			for(int logk = 2; logk <= lvl; ++ logk) {
				test(1 << lvl, 1 << logk, cout);
			}
		}
	}
	return 0;
}
