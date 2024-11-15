#include "../cudaUtils.hpp"

#define NBTHREADS 512


__global__ void dot(float *a, float *b, float *c, int N) {
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
    int N =  datasize / nb_thread;
    float * d_a, * d_b, *d_c;

    //malloc
    checkCudaErrors( cudaMalloc((void**)&d_a, datasize *  sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&d_b, datasize *  sizeof(float)) );
    checkCudaErrors( cudaMalloc((void**)&d_c, sizeof(float) ) );

    std::cout << "test memory dot product\n";

    for(int nb_block = 1; nb_block < N+1; nb_block*=2){
        int nb_loop = N / nb_block;
        startTimer();
        dot<<<nb_block, nb_thread >>>(d_a, d_b, d_c, N);
        std::cout << nb_block << " blocks and " << nb_loop << " loops => ";
        stopTimer((char*)" ");
    }
    startTimer();


    //free
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}