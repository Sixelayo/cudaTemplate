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


#define N (4*1024*1024)
#define NBTHREADS 512


__global__ void dot(float *a, float *b, float *c) {
    __shared__ float sm[NBTHREADS];
    sm[threadIdx.x] = 0;
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    for (int i=index; i<N; i+=blockDim.x*gridDim.x)
        sm[threadIdx.x] += a[i] * b[i];
    __syncthreads();
    if (threadIdx.x==0) {
        float sum = 0;
        for (int i= 0; i<NBTHREADS; i++)
            sum += sm[i];
        atomicAdd(c, sum);
    }
}


int main(void){
    int datasize = 32*1024*1024;
    int nb_thread = 512;
    int truc =  datasize / nb_thread;
    float * d_a, * d_b, *d_c;

    //malloc
    checkCudaErrors( cudaMalloc((void**)&d_a, datasize *  sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&d_b, datasize *  sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&d_c, sizeof(float) ) );

    for(int nb_block = 1; nb_block < truc+1; nb_block*=2){
        int nb_loop = N / nb_block;
        startTimer();
        dot<<<nb_block, nb_thread >>>(d_a, d_b, d_c);
        std::cout << nb_block << " blocks and " << nb_loop << " loops => ";
        stopTimer((char*)" ");
    }
    startTimer();


    std::cout << "test memory dot product\n";


    //free
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}