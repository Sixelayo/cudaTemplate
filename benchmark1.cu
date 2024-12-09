
//CUDA optimisation benchmark
//this was compile using Visual Studio. Compilation for vscode should use nvcc and all
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>

#include <random>
#include <chrono>
#include <iostream>
#include <cmath>


#define N 16*256*256
#define M 512

#define T 900

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char* file, const int line){
	if (err != cudaSuccess)
	{
		fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
			file, line, (int)err, cudaGetErrorString(err));
		system("pause");
		exit(1);
	}
}
inline void checkKernelError() {
	cudaDeviceSynchronize();
	checkCudaErrors(cudaGetLastError());
}

cudaEvent_t cstart, cstop;
float cdiff;
void startTimer() {
	cudaEventCreate(&cstart);
	cudaEventCreate(&cstop);
	cudaEventRecord(cstart);

}

float stopTimer(char* str){
	cudaEventRecord(cstop);
	cudaEventSynchronize(cstop);
	cudaEventElapsedTime(&cdiff, cstart, cstop);
	printf("%s\tCUDA time is % .2f ms\n",str, cdiff);
	cudaEventDestroy(cstart);
	cudaEventDestroy(cstop);
	return cdiff;
}

__device__ float next(float xn) {
	return 4 * xn * (1 - xn);
}


__global__ void add(float* a, float* b, float* c, int n) {
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	//expensive computation for benchmark
	float xn = a[index];
	for (int i = 0; i < T; i++) {
		xn = next(xn);
	}
	//if (index < n) c[index] = a[index] * b[index];
	
	//printf("var : %d\n", threadIdx); //printf in GPU
	//CUDA Runtime API error 700: an illegal memory access was encountered.
	/*
	float zero = 0.0;
	float* zeroptr = (float*) ((int) zero);
	*(float*) zeroptr = 3.0; //kuda kernel error : nullptr deref
	*/
}


float comparef(float* v1, float* v2, int n) {
	double diff = 0.0;
	for (int i = 0; i < n; i++) {
		diff += (v1[i] - v2[i]) * (v1[i] - v2[i]);
	}
	return (float)(sqrt(diff / (double)n));
}

void random_vector(float* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = (float)(std::rand() % 1000000) / 1000000;
	}
}
void setAllToOne(float* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = 1;
	}
}

void checkAllEqual(float* a, int vecSize, float value) {
	bool error = false;
	for (int i = 0; i < vecSize; i++) {
		if (abs(a[i] - value) > 0.01) {
			error = true;
			std::cout << a[i] << " / ";
		}
	}
	if (error) {
		std::cout << "\tnot all are equal to " << value;
	}
	else {
		std::cout << "\tall values are equal to " << value;
	}
	std::cout << "\n";
}

void golden_model(float* a, float* b, float* c, int n) {
	for (int i = 0; i < n; i++) {
		c[i] = a[i] * b[i];
	}
}


__global__ void warp_ok(float* d_idata, float* d_odata) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = d_idata[i];
	if ((threadIdx.x / 32) % 2 == 0) {
		for (int j = 0; j < 10000; j++) x += 0.1;
		d_odata[i] = x;
	}
	else {
		for (int j = 0; j < 10000; j++) x += 0.2;
		d_odata[i] = x; 
	}
}
__global__ void warp_ko(float* d_idata, float* d_odata) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float x = d_idata[i];
	if (threadIdx.x % 2 == 0) {
		for (int j = 0; j < 10000; j++) x += 0.1;
		d_odata[i] = x;
	}
	else {
		for (int j = 0; j < 10000; j++) x += 0.2;
		d_odata[i] = x;
	}
}

__global__ void coalesce(float* d_idata, float* d_odata, int datasize, int stride) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = (stride * i) % datasize;
	d_odata[j] = d_idata[j];
}

__global__ void bank(float* d_idata, float* d_odata, int stride)
{
	__shared__ double sm[512]; // double amplifies the conflicts
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	sm[threadIdx.x] = d_idata[i];
	__syncthreads();
	int n = (stride * threadIdx.x) % 512;
	d_odata[i] = sm[n];
}

__global__ void reduceV1(float* g_idata, float* g_odata) {
	extern __shared__ int sdata[512];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		if (tid % (2 * s) == 0) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global memory
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceV2(float* g_idata, float* g_odata) {
	extern __shared__ int sdata[512];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = 1; s < blockDim.x; s *= 2) {
		int index = 2 * s * tid;
		if (index < blockDim.x) {
			sdata[index] += sdata[index + s];
		}
		__syncthreads();
	}
	// write result for this block to global memory
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceV3(float* g_idata, float* g_odata) {
	extern __shared__ int sdata[512];
	// each thread loads one element from global to shared mem
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	sdata[tid] = g_idata[i];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global memory
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceV4(float* g_idata, float* g_odata) {
	extern __shared__ int sdata[512];
	// perform first add of reduction upon reading from
	// global memory and writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
		if (tid < s) {
			sdata[tid] += sdata[tid + s];
		}
		__syncthreads();
	}
	// write result for this block to global memory
	if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void reduceV5(float* g_idata, float* g_odata) {
	volatile extern __shared__ int sdatav5[512];
	// perform first add of reduction upon reading from
	// global memory and writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	sdatav5[tid] = g_idata[i] + g_idata[i + blockDim.x];
	__syncthreads();
	// do reduction in shared mem
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
		if (tid < s)
			sdatav5[tid] += sdatav5[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		for (int s = 32; s > 0; s /= 2)
			sdatav5[tid] += sdatav5[tid + s];
	}
	// write result for this block to global memory
	if (tid == 0) g_odata[blockIdx.x] = sdatav5[0];
}

__global__ void reduceV6(float* g_idata, float* g_odata, unsigned int datasize) {
	volatile extern __shared__ int sdatav6[512];
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
	unsigned int gridsize = (blockDim.x * 2) * gridDim.x;
	sdatav6[tid] = 0;
	while (i < datasize) {
		sdatav6[tid] += g_idata[i] + g_idata[i + blockDim.x];
		i += gridsize;
	}
	__syncthreads();
	for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1)
	{
		if (tid < s)
			sdatav6[tid] += sdatav6[tid + s];
		__syncthreads();
	}
	if (tid < 32)
	{
		for (int s = 32; s > 0; s /= 2)
			sdatav6[tid] += sdatav6[tid + s];
	}
	if (tid == 0) g_odata[blockIdx.x] = sdatav6[0];
}


int main(void) {
	std::cout << "Starting program ...\n";

	//allocatoin
	float* a, * b, * c, * reducResult, * d_a, * d_b, * d_c, *d_reducResult;
	int size = N * sizeof(float);
	cudaMallocHost((void**)&a,size); random_vector(a, N);
	cudaMallocHost((void**)&b,size); random_vector(b, N);
	cudaMallocHost((void**)&c,size); random_vector(c, N);

	checkCudaErrors( cudaMalloc((void**)&d_a, size) );
	checkCudaErrors( cudaMalloc((void**)&d_b, size) );
	checkCudaErrors( cudaMalloc((void**)&d_c, size) );

	cudaMallocHost((void**)&reducResult, size / M);
	checkCudaErrors( cudaMalloc((void**)&d_reducResult, size / M));


	//float v; //timer


	std::cout << "warp test\n";
	//warp OK
	startTimer();
	checkCudaErrors( cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	warp_ok << <(N + M - 1) / M, M >> > (d_a, d_b);
	checkCudaErrors( cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost));
	stopTimer((char*)"\twarp OK\t");

	//warp not OK
	startTimer();
	checkCudaErrors( cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	warp_ko << <(N + M - 1) / M, M >> > (d_a, d_b);
	checkCudaErrors( cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost));
	stopTimer((char*)"\twarp bad\t");

	//coalescence
	std::cout << "\n\nCoalescence test : \n";
	for (int stride = 1; stride < 16; stride++) {
		std::cout << "\tstride : " << stride <<"\t";
		checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
		startTimer();
		coalesce << <(N + M - 1) / M, M >> > (d_a, d_b, N, stride);
		stopTimer((char*)"\t");
		checkCudaErrors(cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost));
	}

	//bank test
	std::cout << "\n\nBank conflict test : \n";
	for (int stride = 1; stride < 40; stride++) {
		std::cout << "\tstride : " << stride << "\t";
		checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
		startTimer();
		bank << <(N + M - 1) / M, M >> > (d_a, d_b, stride);
		stopTimer((char*)"\t");
		checkCudaErrors(cudaMemcpy(b, d_b, size, cudaMemcpyDeviceToHost));
	}

	//reduction test
	std::cout << "\n\nReduction algorythm test v1: \n";
	setAllToOne(a, N);
	checkAllEqual(a, N, 1);
	checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	startTimer();
	reduceV1 << <(N + M - 1) / M, M >> > (d_a, d_reducResult);
	stopTimer("\tReduction time :");
	checkCudaErrors(cudaMemcpy(reducResult, d_reducResult, size/M, cudaMemcpyDeviceToHost));
	checkAllEqual(reducResult, N / M, 512);

	//reduciton test v2 without branching in a warp
	//Warp doesn't run in parralel, thus we can save time by indexing reindexing thread so we have "useless warp" that does purely nothing (in prior version we had at least 1 thread working per wrap)
	std::cout << "\n\nReduction algorythm test v2: \n";
	setAllToOne(a, N);
	checkAllEqual(a, N, 1);
	checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	startTimer();
	reduceV2 << <(N + M - 1) / M, M >> > (d_a, d_reducResult);
	stopTimer("\tReduction time :");
	checkCudaErrors(cudaMemcpy(reducResult, d_reducResult, size / M, cudaMemcpyDeviceToHost));
	checkAllEqual(reducResult, N / M, 512);

	//reduction test v3. Improvement because of bank conflict
	std::cout << "\n\nReduction algorythm test v3: \n";
	setAllToOne(a, N);
	checkAllEqual(a, N, 1);
	checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	startTimer();
	reduceV3 << <(N + M - 1) / M, M >> > (d_a, d_reducResult);
	stopTimer("\tReduction time :");
	checkCudaErrors(cudaMemcpy(reducResult, d_reducResult, size / M, cudaMemcpyDeviceToHost));
	checkAllEqual(reducResult, N / M, 512);


	//reduction test v4. Improvement because half thread useless at lauchn
	//don't forgot to halve the number of block (ie thread)
	std::cout << "\n\nReduction algorythm test v4: \n";
	setAllToOne(a, N);
	checkAllEqual(a, N, 1);
	checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	startTimer();
	reduceV4 << <(N + M - 1) / (M*2), M >> > (d_a, d_reducResult);
	stopTimer("\tReduction time :");
	checkCudaErrors(cudaMemcpy(reducResult, d_reducResult, size / M, cudaMemcpyDeviceToHost));
	checkAllEqual(reducResult, N / (2*M), 1024);

	//reduction test v5. Improvement because half thread useless at lauchn
	//don't forgot to halve the number of block (ie thread)
	std::cout << "\n\nReduction algorythm test v5: \n";
	setAllToOne(a, N);
	checkAllEqual(a, N, 1);
	checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
	startTimer();
	reduceV5 << <(N + M - 1) / (M * 2), M >> > (d_a, d_reducResult);
	stopTimer("\tReduction time :");
	checkCudaErrors(cudaMemcpy(reducResult, d_reducResult, size / M, cudaMemcpyDeviceToHost));
	checkAllEqual(reducResult, N / (2 * M), 1024);

	//reduction V6 : less trhead more work per thread
	for (int loadwork = 2; loadwork < 65; loadwork *= 2) {
		std::cout << "\n\nReduction algorythm test v6: \n";
		std::cout << "with loadwork = " << loadwork << "\n";
		setAllToOne(a, N);
		checkAllEqual(a, N, 1);
		checkCudaErrors(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
		startTimer();
		reduceV6 << <(N + M - 1) / (M * loadwork), M >> > (d_a, d_reducResult, N);
		stopTimer("\tReduction time :");
		checkCudaErrors(cudaMemcpy(reducResult, d_reducResult, size / M, cudaMemcpyDeviceToHost));
		checkAllEqual(reducResult, N / (loadwork * M), 512*loadwork);
	}


	/*
	float v; //used for bandwidth test

	//CUDA
	startTimer();
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	v = stopTimer("H2D");
	float bwh2d = N * sizeof(float) / (pow(1000, 2) * v);
	checkCudaErrors( cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice) );
	printf("\tbandwitdh h2d : %.2f\n", bwh2d);

	startTimer();
	add << <(N + M - 1) / M, M >> > (d_a, d_b, d_c, N);
	v = stopTimer("kernel");
	checkKernelError();
	//float bwd2d = 3 * N * sizeof(float) / (pow(1000, 2)*v); //3 array thus 3 copy
	float glfops = 3.0f * (float) N * (float)T / (v*1000000); //3 array thus 3 copy
	//printf("\tbandwidth d2d : %.2f\n", bwd2d);
	printf("\Gflops d2d : %.2f\n", glfops);

	startTimer();
	checkCudaErrors( cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost) );
	v = stopTimer("D2H");
	float bwd2h = N * sizeof(float) / (pow(1000, 2) * v);
	printf("\tbandwitdh d2h : %.2f\n", bwd2h);

	//CPU
	golden_model(a, b, c2, N);


	float rmsd = comparef(c, c2, N);
	std::cout << "Root mean square deviation : " << rmsd << "\n";
	*/
	
	//free memory
	cudaFreeHost(a); cudaFreeHost(b); cudaFreeHost(c); cudaFreeHost(reducResult);
	checkCudaErrors( cudaFree(d_a) );
	checkCudaErrors( cudaFree(d_b) );
	checkCudaErrors( cudaFree(d_c) );
	checkCudaErrors( cudaFree(d_reducResult) );
	return 0;
}