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

int main() {
	ofstream fout("result.txt");

	for(int i=0; i<10; ++i) {
		int size = 1 << 20, k = 1 << 2;
		Data* input = generateRandomInput(size);
		//cout << input->get_data() << endl;

		clock_t start = clock();
		Result res1 = serial_kmers(*input, k);
		cout << size << "\t" << k << "\tcpu time: " << (clock() - start) << endl;

		start = clock();
		Result res2 = parallel_kmers(*input, k);
		cudaDeviceSynchronize();
		cout << size << "\t" << k << "\tgpu time: " << (clock() - start) << endl;


		cout << res1.equal(res2) << endl;
	}
	return 0;
}
