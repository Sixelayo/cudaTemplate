#include "../cudaUtils.hpp"


#define SIZE 4096
#define N 1024*1024*32
#define M 128


__constant__ float d_idata_const[SIZE];
//float* h_idata; not needed
float* d_idata;
float* d_odata;

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