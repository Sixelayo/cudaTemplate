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

/*
Initialize n values of a with a random float
*/
void random_float_vector(float* a, int n) {
	for (int i = 0; i < n; i++) {
		a[i] = (float)(std::rand() % 1000000) / 1000000;
	}
}

//used by teacher implementation
float* randomArray(int size, float min, float max){
    float* array = (float*)malloc(size*sizeof(float));
    for (int i = 0; i < size; i++) {
		array[i] = min+(max-min)*((float)(std::rand() % 1000000) / 1000000);
	}
    return array;
}


float comparef (float* a, float* b, int n) {
    double diff = 0.0;
    for (int i=0;i<n;i++)
        diff += (a[i]-b[i])*(a[i]-b[i]);
    return (float)(sqrt(diff/(double)n));
}
