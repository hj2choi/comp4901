/*
 created by xjia@ust.hk 2015/04/15
 revised by Lipeng WANG lwangay@connect.ust.hk, NOV. 11 2016
 */
#include <iostream>
#include <cstdio>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <unistd.h>
#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
using namespace std;

const int numBits = 6;
const int numPart = 1 << numBits;
const int mask = (1 << numBits) - 1;
const int numThreads = 128;
const int numBlocks = 512;

/*
 return the partition ID of the input element
 */
__host__ __device__
int getPartID(int element) {
	//element >>= (totalBits - numBits);
	return element & mask;
}

/*
 input: d_key[], array size N
 output: d_pixArray[]
 function: for input array d_key[] with size N, return the partition ID array d_pixArray[]
 */
__global__
void mapPart(int d_pidArray[], int d_key[], int N) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNumber = blockDim.x * gridDim.x;

	while (tid < N) {
		d_pidArray[tid] = getPartID(d_key[tid]);
		tid += threadNumber;
	}
}

/*
 input: d_pidArray[], array size N
 output: d_Hist[]
 function: calculate the histogram d_Hist[] based on the partition ID array d_pidArray[]
 */
__global__
void count_Hist(int d_Hist[], int d_pidArray[], int N) {
	__shared__ int s_Hist[numThreads * numPart];
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;
	int threadNumber = blockDim.x * gridDim.x;
	int offset = threadIdx.x * numPart;

	for (int i = 0; i < numPart; ++i)
		s_Hist[i + offset] = 0;

	for (int i = threadId; i < N; i += threadNumber)
		s_Hist[offset + d_pidArray[i]]++;

	for (int i = 0; i < numPart; ++i)
		d_Hist[i * threadNumber + threadId] = s_Hist[offset + i];
	__syncthreads();
}
/*
 input: d_pidArray[] (partition ID array), d_psSum[] (prefix sum of histogram), array size N
 output: d_loc[] (location array)
 function: for each element, calculate its corresponding location in the result array based on its partition ID and prefix sum of histogram
 */
__global__
void write_Hist(int d_loc[], int d_pidArray[], int d_psSum[], int N) {
	__shared__ int s_psSum[numThreads * numPart];
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNumber = gridDim.x * blockDim.x;
	int offset = threadIdx.x * numPart;

	for (int i = 0; i < numPart; ++i)
		s_psSum[i + offset] = d_psSum[threadId + i * threadNumber];

	for (int i = threadId; i < N; i += threadNumber) {
		int pid = d_pidArray[i];
		d_loc[i] = s_psSum[pid + offset];
		s_psSum[pid + offset]++;
	}
}

/*
 input: d_psSum[] (prefix sum of histogram), array size N
 output: d_startPos[]  start position of each partition
 function: for each partition (chunk to be loaded in the join step), calculate its start position in the result array (the first element's position of this partition)
 */
__global__
void getStartPos(int d_startPos[], int d_psSum[], int N) {
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNumber = gridDim.x * blockDim.x;

	if (tid >= numPart)
		return;
	d_startPos[tid] = d_psSum[tid * threadNumber];
}

/*
 input: d_key[],d_value[],d_loc[],array size []
 output: out_key[],out_value[]
 function: rewrite the (key,value) pair to its corresponding position based on location array d_loc[]
 */
__global__
void scatter(int out_key[], float out_value[], int d_key[], float d_value[],
		int d_loc[], int N) {
	int threadId = threadIdx.x + blockIdx.x * blockDim.x;
	int threadNumber = blockDim.x * gridDim.x;

	while (threadId < N) {
		out_key[d_loc[threadId]] = d_key[threadId];
		out_value[d_loc[threadId]] = d_value[threadId];
		threadId += threadNumber;
	}
}

/*
 function: split the (key,value) array with size N, record the start position of each partition at the same time
 */
void split(int *d_key, float *d_value, int *d_startPos, int N) {
	int *d_keyOut, *d_pidArray, *d_loc, *d_Hist, *d_psSum;
	float *d_valueOut;
	dim3 grid(numBlocks);
	dim3 block(numThreads);
	int totalThreads = grid.x * block.x;
	int Hist_len = totalThreads * numPart;

	/* add your code here */
	cudaMalloc(&d_pidArray, sizeof(int) * N);
	cudaMalloc(&d_Hist, sizeof(int) * Hist_len);
	cudaMalloc(&d_psSum, sizeof(int) * Hist_len);
	cudaMalloc(&d_loc, sizeof(int) * N);
	cudaMalloc(&d_keyOut, sizeof(int) * N);
	cudaMalloc(&d_valueOut, sizeof(float) * N);

	mapPart<<<grid, block>>>(d_pidArray, d_key, N);
	count_Hist<<<grid, block>>>(d_Hist, d_pidArray, N);

	thrust::device_ptr<int> dev_Hist(d_Hist);
	thrust::device_ptr<int> dev_psSum(d_psSum);
	thrust::exclusive_scan(dev_Hist, dev_Hist + Hist_len, dev_psSum);

	write_Hist<<<grid, block>>>(d_loc, d_pidArray, d_psSum, N);
	getStartPos<<<grid, block>>>(d_startPos, d_psSum, N);
	scatter<<<grid, block>>>(d_keyOut, d_valueOut, d_key, d_value, d_loc, N);

	cudaMemcpy(d_key, d_keyOut, sizeof(int) * N, cudaMemcpyDeviceToDevice);
	cudaMemcpy(d_value, d_valueOut, sizeof(float) * N,
			cudaMemcpyDeviceToDevice);

	cudaFree(d_pidArray);
	cudaFree(d_Hist);
	cudaFree(d_psSum);
	cudaFree(d_loc);
	cudaFree(d_keyOut);
	cudaFree(d_valueOut);
}

void invoke_split(int *d_result, int *d_key1,
		float *d_value1, int *d_key2, float *d_value2, int N1, int N2) {
	int *d_startPos1, *d_startPos2, *h_startPos1, *h_startPos2, *h_key1, *h_key2;

	h_startPos1 = (int*) malloc(sizeof(int) * numPart);
	h_startPos2 = (int*) malloc(sizeof(int) * numPart);
	h_key1 = (int*) malloc(sizeof(int) * N1);
	h_key2 = (int*) malloc(sizeof(int) * N2);

	cudaMalloc(&d_startPos1, sizeof(int) * numPart);
	cudaMalloc(&d_startPos2, sizeof(int) * numPart);
	split(d_key1, d_value1, d_startPos1, N1);
	split(d_key2, d_value2, d_startPos2, N2);

	cudaMemcpy(h_startPos1, d_startPos1, sizeof(int)*numPart, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_startPos2, d_startPos2, sizeof(int)*numPart, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_key1, d_key1, sizeof(int)*N1, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_key2, d_key2, sizeof(int)*N2, cudaMemcpyDeviceToHost);

	//check results
	for(int i = 0; i < numPart; i++){
		int start_of_this_bucket = h_startPos1[i];
		int end_of_this_bucket = -1;
		if(i == numPart-1){
			end_of_this_bucket = N1;
		}else{
			end_of_this_bucket = h_startPos1[i + 1];
		}
		for(int j = start_of_this_bucket; j < end_of_this_bucket; j++){
			if(getPartID(h_key1[j]) != i){
				fprintf(stderr, "Error! h_key1[%d] = %d should be in bucket %d not %d.\n", j, h_key1[j], getPartID(h_key1[j]), i);
				exit(-1);
			}
		}
	}
	for(int i = 0; i < numPart; i++){
			int start_of_this_bucket = h_startPos2[i];
			int end_of_this_bucket = -1;
			if(i == numPart-1){
				end_of_this_bucket = N2;
			}else{
				end_of_this_bucket = h_startPos2[i + 1];
			}
			for(int j = start_of_this_bucket; j < end_of_this_bucket; j++){
				if(getPartID(h_key2[j]) != i){
					fprintf(stderr, "Error! h_key2[%d] = %d should be in bucket %d not %d.\n", j, h_key2[j], getPartID(h_key2[j]), i);
					exit(-1);
				}
			}
		}

	cudaFree(d_startPos1);
	cudaFree(d_startPos2);

	free(h_startPos1);
	free(h_startPos2);
	free(h_key1);
	free(h_key2);

}
void makedata(int N1, int N2, vector<int> &key1, vector<float> &value1,
		vector<int> &key2, vector<float> &value2) {
	std::srand(time(NULL));
	key1.resize(N1);
	value1.resize(N1);
	key2.resize(N2);
	value2.resize(N2);
	for (int i = 0; i < N1; ++i) {
		key1[i] = i;
		value1[i] = i * 1.0;
	}

	for (int i = 0; i < N2; ++i) {
		key2[i] = i;
		value2[i] = i * 2.0;
	}
	random_shuffle(key1.begin(), key1.end());
	random_shuffle(key2.begin(), key2.end());
	random_shuffle(value1.begin(), value1.end());
	random_shuffle(value2.begin(), value2.end());
}
int main(int argc, char** argv) {
	int *h_key1, *h_key2, *d_key1, *d_key2;
	float *h_value1, *h_value2, *d_value1, *d_value2;
	int *h_result, *d_result;
	int N1, N2;

	std::vector<int> k1, k2;
	std::vector<float> v1, v2;
	if (argc == 1) {
		N1 = 400000;
		N2 = 300000;
		fprintf(stderr, "Generating random input data...\n");
		//make random input data
		makedata(N1, N2, k1, v1, k2, v2);
	} else {
		//read from a file
		if (access(argv[1], F_OK | R_OK) != 0) {
			fprintf(stderr, "Error while reading input file \'%s\'\n", argv[1]);
			return -1;
		}
		fprintf(stderr, "Reading from input file \'%s\'\n", argv[1]);
		freopen(argv[1], "r", stdin);
		scanf("%d%d", &N1, &N2);
		k1.resize(N1);
		v1.resize(N1);
		k2.resize(N2);
		v2.resize(N2);
		int key;
		float value;
		for (int i = 0; i < N1; i++) {
			scanf("%d%f", &key, &value);
			k1[i] = key;
			v1[i] = value;
		}
		for (int i = 0; i < N2; i++) {
			scanf("%d%f", &key, &value);
			k2[i] = key;
			v2[i] = value;
		}
	}
	assert(N1 < 524288);
	assert(N2 < 524288);

	h_key1 = (int*) malloc(N1 * sizeof(int));
	h_key2 = (int*) malloc(N2 * sizeof(int));
	h_value1 = (float*) malloc(N1 * sizeof(float));
	h_value2 = (float*) malloc(N2 * sizeof(float));
	h_result = (int*) malloc(N1 * sizeof(int));

	cudaMalloc(&d_key1, N1 * sizeof(int));
	cudaMalloc(&d_key2, N2 * sizeof(int));
	cudaMalloc(&d_value1, N1 * sizeof(float));
	cudaMalloc(&d_value2, N2 * sizeof(float));
	cudaMalloc(&d_result, N1 * sizeof(int));

	for (int i = 0; i < N1; ++i) {
		h_key1[i] = k1[i];
		h_value1[i] = v1[i];
	}
	for (int i = 0; i < N2; ++i) {
		h_key2[i] = k2[i];
		h_value2[i] = v2[i];
	}

	memset(h_result, -1, sizeof(int) * N1);
	cudaMemcpy(d_key1, h_key1, sizeof(int) * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, h_result, sizeof(int) * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_key2, h_key2, sizeof(int) * N2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_value1, h_value1, sizeof(float) * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_value2, h_value2, sizeof(float) * N2, cudaMemcpyHostToDevice);

	invoke_split(d_result, d_key1, d_value1, d_key2,
			d_value2, N1, N2);

	fprintf(stderr, "success! \n");

	free(h_key1);
	free(h_key2);
	free(h_value1);
	free(h_value2);
	free(h_result);

	cudaFree(d_key1);
	cudaFree(d_key2);
	cudaFree(d_value1);
	cudaFree(d_value2);
	cudaFree(d_result);

	cudaDeviceReset();
	return 0;
}
