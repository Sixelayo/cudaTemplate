#include "../cudaUtils.hpp"

#define TILE_WIDTH 16

__global__ void matrixMul_GPU(float* Md, float* Nd, float* Pd, int width)
{
    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    int i = by * TILE_WIDTH + ty;
    int j = bx * TILE_WIDTH + tx;

    float sum = 0;
    for (int k = 0; k < width; ++k)
    {
    float a = Md[i*width + k];
    float b = Nd[k*width + j];
    sum += a * b;
    }
    Pd[i * width + j] = sum;
}

__global__ void matrixMul_GPUshared(float* Md, float* Nd, float* Pd, int Width)
{
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x; int by = blockIdx.y;
  int tx = threadIdx.x; int ty = threadIdx.y;

  // identify the row and column of the Pd element to work on
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  if (Row<Width && Col<Width) {
 
      float Pvalue = 0;
      // loop over the Md and Nd tiles required to compute the Pd element
      for (int m = 0; m < Width/TILE_WIDTH; ++m)
      {
        // collaborative loading of Md and Nd tiles into shared memory
        Mds[ty][tx] = Md[Row*Width + (m*TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[Col + (m*TILE_WIDTH + ty)*Width];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
          Pvalue += Mds[ty][k] * Nds[k][tx];
        __syncthreads();
       }
       Pd[Row*Width+Col] = Pvalue;
    }
}

// CPU version of matrix multiplication
void matrixMul_CPU(float* M, float* N, float* P, int Width)
{
  for (int i = 0; i < Width; ++i)
    for (int j = 0; j < Width; ++j)
    {
      float sum = 0;
      for (int k = 0; k < Width; ++k)
      {
        float a = M[i*Width + k];
        float b = N[k*Width + j];
        sum += a * b;
      }
      P[i * Width + j] = sum;
    }
}

void testMatrixmul() {
    printf("\nMatrixmul test\n");
    printf("---------------------\n");

    // StopWatchInterface *timer = 0;
    // cudaEvent_t cstart, cstop;
    float t;
    
    unsigned int width = 512;
    printf("matrix size: %d x %d\n", width, width);
    unsigned int datasize = width*width;
    unsigned int memsize = sizeof(float) * datasize;
    float* h_M = randomArray(datasize, -0.5, 0.5);
    float* h_N = randomArray(datasize, -0.5, 0.5);
    float* h_P = randomArray(datasize, 0, 0);
    float* h_golden = randomArray(datasize, 0, 0);
    // allocate device memory
    float* d_M, *d_N, *d_P;
    checkCudaErrors(cudaMalloc((void**)&d_M, memsize));
    checkCudaErrors(cudaMalloc((void**)&d_N, memsize));
    checkCudaErrors(cudaMalloc((void**)&d_P, memsize));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_M, h_M, memsize, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_N, h_N, memsize, cudaMemcpyHostToDevice));

    
    startTimer();
    matrixMul_CPU(h_M, h_N, h_golden, width);
    t=stopTimer((char*)"\t\tfoo");
    printf("CPU: %.2f ms\n", t);

    startTimer();
    matrixMul_GPU << <dim3(width / TILE_WIDTH, width / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH) >> > (d_M, d_N, d_P, width);
    t = stopTimer((char*)"\t\tfoo");
    printf("GPU: %.2f ms, ", t);
    checkCudaErrors(cudaMemcpy(h_P, d_P, memsize, cudaMemcpyDeviceToHost));
    printf("RMSE: %.2f\n",comparef(h_golden,h_P,datasize));

    startTimer();
    matrixMul_GPUshared << <dim3(width / TILE_WIDTH, width / TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH) >> > (d_M, d_N, d_P, width);
    t = stopTimer((char*)"\t\tfoo");
    printf("GPU shared: %.2f ms, ", t);
    checkCudaErrors(cudaMemcpy(h_P, d_P, memsize, cudaMemcpyDeviceToHost));
    printf("RMSE: %.2f\n",comparef(h_golden,h_P,datasize));

    // cleanup memory
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_golden);
    checkCudaErrors(cudaFree(d_M));
    checkCudaErrors(cudaFree(d_N));
    checkCudaErrors(cudaFree(d_P));
    

}

int main(){
    testMatrixmul();
}