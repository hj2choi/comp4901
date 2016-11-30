/*
 * ITSC account: hjchoi
 * Name: CHOI, Hong Joon
 * Student id: 20161472
 */

/*
 skeleton code for assignment3 COMP4901D
 Hash Join
 xjia@ust.hk 2015/04/15
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
const int totalBits = 19;
const int numPart = 1 << numBits;
const int numPerPart = 1 << (totalBits - numBits);
const int mask = (1 << numBits) - 1;
const int numThreads = 128;
const int numBlocks = 512;

void join_cpu(int *key1, float *value1, int *key2, float *value2,int N1,int N2,int *result){
    for(int i = 0; i < N1; i++){
        result[i] = -1;
        for(int j = 0; j < N2; j++){
            if(key1[i] == key2[j]){
                result[i] = j;
            }
        }
    }
}

bool check(int *res1, int *res2, int N){
    for(int i = 0; i < N; i++){
        if(res1[i] != res2[i]){
            printf("Wrong! res1[%d]: %d, res2[%d]: %d\n", i, res1[i], i, res2[i]);
            //return false;
        }
    }
    return true;
}

/*
 return the partition ID of the input element
 */
//TODO
__device__
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

/*
 function: perform hash join on two (key,value) arrays
 */
__global__
void join(int d_result[], int d_key1[], float d_value1[], int d_key2[],
		float d_value2[], int d_startPos1[], int d_startPos2[], int N1,
		int N2) {
	__shared__ int s_key[numPerPart];
	__shared__ int startPos1, startPos2;
	__shared__ int endPos1, endPos2;
	__shared__ int numOfThisPart1, numOfThisPart2;

	/* add your code here */
	/*
		Oh my dear heterogeneous god,
		Do not let my code fall into sin of random undebuggable bugs
	*/
	int tx = threadIdx.x;
	//int blockIdx = blockDim.x;
	int bx = blockIdx.x;
	//copy each bucket information into shared memory.
	//each block is responsible for each bucket
	if (tx == 0) {
		startPos1 = d_startPos1[bx];
		startPos2 = d_startPos2[bx];
		if (bx+1==gridDim.x) {	// if we are looking at last bucket, endPosition is basically a length of array
			endPos1 = N1;
			endPos2 = N2;
		} else {	// else, endPos is startPos of next bucket
			endPos1 = d_startPos1[bx+1];
			endPos2 = d_startPos2[bx+1];
		}
		numOfThisPart1 = endPos1 - startPos1;
		numOfThisPart2 = endPos2 - startPos2;

    // load each buckets of array 2 into shared memory and synchronize
		for (int i=0; i<numOfThisPart2; ++i) {
			s_key[i] = d_key2[startPos2+i];
		}
	}
	__syncthreads();
	// each thread is now responsible for each element in the bucket.
	if (tx>=numOfThisPart1) {
		return;
	}
	int result_tmp = -1;
	int d_key_tmp = d_key1[startPos1+tx];

	// for each element in bucket 1, search for matching element in bucket 2
	for (int i=0; i<numOfThisPart2; ++i) {
		if (d_key_tmp == s_key[i]) {
			result_tmp = startPos2+i;
		}
	}
	d_result[startPos1+tx] = result_tmp;

}

void cpu_hashJoin(int *d_result, int *d_key1, float *d_value1, int *d_key2,
		float *d_value2, int N1, int N2) {
  int *d_startPos1, *d_startPos2;
  cudaMalloc(&d_startPos1, sizeof(int) * numPart);
  cudaMalloc(&d_startPos2, sizeof(int) * numPart);
  split(d_key1, d_value1, d_startPos1, N1);
  split(d_key2, d_value2, d_startPos2, N2);


  printf("executing hash join on cpu....");

  // go through each bucket
  /*for (int i=0; i<numPart; ++i) {

  }*/
}

void hashJoin(int *d_result, int *d_key1, float *d_value1, int *d_key2,
		float *d_value2, int N1, int N2) {
	int *d_startPos1, *d_startPos2;
	cudaMalloc(&d_startPos1, sizeof(int) * numPart);
	cudaMalloc(&d_startPos2, sizeof(int) * numPart);
	split(d_key1, d_value1, d_startPos1, N1);
	split(d_key2, d_value2, d_startPos2, N2);

	dim3 grid(numPart);
	dim3 block(1024);

	join<<<grid,block>>>(d_result, d_key1, d_value1, d_key2, d_value2, d_startPos1, d_startPos2, N1, N2);
}
void makedata(int N1, int N2, vector<int> &key1, vector<float> &value1,
		vector<int> &key2, vector<float> &value2) {
	std::srand(0);
	key1.resize(N1);value1.resize(N1);key2.resize(N2);value2.resize(N2);
	for(int i = 0; i < N1; ++i) {
		key1[i] = i;
		value1[i] = i * 1.0;
	}

	for(int i = 0; i < N2; ++i) {
		key2[i] = i;
		value2[i] = i * 2.0;
	}
	random_shuffle(key1.begin(),key1.end());
	random_shuffle(key2.begin(),key2.end());
	random_shuffle(value1.begin(),value1.end());
	random_shuffle(value2.begin(),value2.end());
}
int main(int argc, char** argv) {
	int *h_key1, *h_key2, *d_key1, *d_key2;
	float *h_value1, *h_value2, *d_value1, *d_value2;
	int *h_result, *d_result;
	int N1, N2;
	//int *h_result_base;

	std::vector<int> k1, k2;
	std::vector<float> v1, v2;
	if (argc == 1) {
		N1 = 400000;
		N2 = 300000;
		fprintf(stderr, "Generating default input data...\n");
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
	//h_result_base = (int*)malloc(N1 * sizeof(int));

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

	printf("gpu_join()...");
	memset(h_result, -1, sizeof(int) * N1);
	cudaMemcpy(d_key1, h_key1, sizeof(int) * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_result, h_result, sizeof(int) * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_key2, h_key2, sizeof(int) * N2, cudaMemcpyHostToDevice);
	cudaMemcpy(d_value1, h_value1, sizeof(float) * N1, cudaMemcpyHostToDevice);
	cudaMemcpy(d_value2, h_value2, sizeof(float) * N2, cudaMemcpyHostToDevice);

	hashJoin(d_result, d_key1, d_value1, d_key2, d_value2, N1, N2);

	cudaMemcpy(h_result, d_result, sizeof(int) * N1, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_key1, d_key1, sizeof(int) * N1, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_key2, d_key2, sizeof(int) * N2, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_value1, d_value1, sizeof(float) * N1, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_value2, d_value2, sizeof(float) * N2, cudaMemcpyDeviceToHost);

	int matched = 0;
	freopen("out.txt", "w", stdout);
	for (int i = 0; i < N1; ++i) {
		if (h_result[i] == -1)
			continue;
		matched++;
		printf("Key %d\nValue1 %.2f Value2 %.2f\n\n", h_key1[i], h_value1[i],
				h_value2[h_result[i]]);
	}
	printf("Matched %d\n", matched);
	fclose(stdout);

	fprintf(stderr, "Matched %d\n", matched);

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
