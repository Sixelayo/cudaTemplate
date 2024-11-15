#include "../cudaUtils.hpp"



/*
Matrix product :
P = M * N of size WIDTH x WIDTH
*/

#define WIDTH 512

// CPU version of matrix multiplication
void CPUmatrixMul(float* M, float* N, float* P, int Width){
    for (int i = 0; i < Width; ++i){
        for (int j = 0; j < Width; ++j){
            float sum = 0;
            for (int k = 0; k < Width; ++k){
                float a = M[i*Width + k];
                float b = N[k*Width + j];
                sum += a * b;
            }
            P[i * Width + j] = sum;
        }
    }  
}

__global__ void matrixMul(float* M, float* N, float* P, int Width){

}


int main(){
    float* h_A, * h_B, *h_C;
    float* d_A, * d_B, *d_C;
    int memsize = WIDTH * WIDTH * sizeof(float);



    //should use cudaMallocHost
    h_A = (float*) malloc(memsize);
    h_B = (float*) malloc(memsize);
    h_C = (float*) malloc(memsize);

    checkCudaErrors( cudaMalloc((void**)&d_A, memsize) );
    checkCudaErrors( cudaMalloc((void**)&d_B, memsize) );
    checkCudaErrors( cudaMalloc((void**)&d_C, memsize) );
}