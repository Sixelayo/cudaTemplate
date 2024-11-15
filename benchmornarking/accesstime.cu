#include <iostream>
#include <stdio.h>

#ifndef NDEBUG
    #define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
    inline void __checkCudaErrors(cudaError err, const char* file, const int line) {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
            system("pause");
            exit(1);
        }
    }
    inline void checkKernelErrors() {
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }
#else //disable GPU debuging
    #define checkCudaErrors(err) err
    inline void checkKernelErrors() {}
#endif


#define SIZE 4096
#define N 1024*1024*32
#define M 128


__constant__ float d_idata_const[SIZE];
//float* h_idata; not needed
float* d_idata;
float* d_odata;

cudaEvent_t cstart, cstop;
float cdiff;
void startTimer() {
	checkCudaErrors( cudaEventCreate(&cstart) );
	checkCudaErrors( cudaEventCreate(&cstop) );
	checkCudaErrors( cudaEventRecord(cstart) );

}
float stopTimer(char* str){
	checkCudaErrors( cudaEventRecord(cstop) );
	checkCudaErrors( cudaEventSynchronize(cstop) );
	checkCudaErrors( cudaEventElapsedTime(&cdiff, cstart, cstop) );
	printf("%s\tCUDA time is % .6f ms\n",str, cdiff);
	checkCudaErrors( cudaEventDestroy(cstart) );
	checkCudaErrors( cudaEventDestroy(cstop) );
	return cdiff;
}


__global__ void testglobalmem(float* d_idata, float* d_odata) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x = 0;
    for (int j = 0; j<SIZE; j++) x += d_idata[j];
    d_odata[i] = x;
}

__global__ void testconstmem(float* d_odata) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x = 0;
    for (int j = 0; j<SIZE; j++) x += d_idata_const[j];
    d_odata[i] = x;
}

__global__ void testregister(float f, float* d_odata) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    float x = 0;
    for (int j = 0; j<SIZE; j++) x += f;
    d_odata[i] = x;
}



int main(void){
    //malloc
    checkCudaErrors( cudaMalloc((void**)&d_idata, SIZE *  sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&d_odata, N* sizeof(float)) );

    std::cout << "test memory access\n";

    startTimer();
    testglobalmem<<<(N + M -1) / M, M>>>(d_idata, d_odata),
    checkKernelErrors();
    stopTimer((char*)"global memory : ");

    startTimer(),
    testconstmem<<<(N + M -1) / M, M>>> (d_odata);
    checkKernelErrors();
    stopTimer((char*)"constant memory : ");


    startTimer();
    testregister<<<(N + M -1) / M, M>>> (1.0f, d_odata);
    checkKernelErrors();
    stopTimer((char*)"register memory : ");

    //free
    cudaFree(d_idata);
    cudaFree(d_odata);
}