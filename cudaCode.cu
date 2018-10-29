#include "cudaCode.h"
#define NUMBER_OF_THREADS 64

namespace cr {
__global__
void addrow(int n, float *x, float *y) {
	__shared__ float yout[NUMBER_OF_THREADS];
	int startLoc ;
	int th, bl;
	th = threadIdx.x;
	bl = blockIdx.x;

	float loc = (n * th )/ NUMBER_OF_THREADS;
	startLoc = bl*n + int(loc);
	float ytemp;
	yout[threadIdx.x] = 0.0f;
	for(int i = 0; i < (n/NUMBER_OF_THREADS); i ++) {
		ytemp  = ytemp + x[startLoc + i];
	}
	yout[th] = ytemp;
	__syncthreads();

	if(th == 1) {
		float yi ;
		for(int i = 0; i < NUMBER_OF_THREADS; i ++) 
			yi += yout[i];
	
		y[bl] = yi;
	}
	__syncthreads();
}

/*
 * This function reduces the matrix into a column vector by summing over
 * all the values. Here, different threads sum over the different elements
 * in a row which are then summed together.
 */
__global__
void reduceColumn(int n, float *x, float *y) {
	__shared__ float yout[NUMBER_OF_THREADS];
	int th, bl;
	th = threadIdx.x;
	bl = blockIdx.x;
	// This reduces the looping. But the time taken seems to be because of
	// the memory

	int startLoc = bl*n + 2*th;
	for(int i = 0; i < n; i+= 2*NUMBER_OF_THREADS) {
		yout[th] += x[startLoc + i] + x[startLoc + i + 1];
	}
	__syncthreads();

	if(th == 1) {
		float yi;
		for(int i = 0; i < NUMBER_OF_THREADS; i ++) 
			yi += yout[i];
		y[bl] = yi;
	}

}

__global__
void addRow2Threads(int n, float *x, float *yout ) {
	float ytemp;
	int startLoc ; //
	int th, bl;
	th = threadIdx.x;
	bl = blockIdx.x;

	startLoc = bl*n + th;

	for(int i = 0; i < n; i+=NUMBER_OF_THREADS) {
		ytemp += x[startLoc + i];
	}
	yout[(bl * NUMBER_OF_THREADS) + th] = ytemp;
	__syncthreads();
}

__global__
void addThreads(int n, float *yout, float *y ) {

	float yi = 0.0f;
	int startAddress = (blockIdx.x*100 + threadIdx.x) * NUMBER_OF_THREADS;
	for(int i = 0; i < NUMBER_OF_THREADS; i ++)
		yi += yout[startAddress + i];

	y[blockIdx.x*100 + threadIdx.x] = yi;
	__syncthreads();
}

void cudaCode::cudaRowReduce(void){

	float milliseconds = 0;

	m = 25000;
	n = 32000;

	N  = m*n;
	x = (float*)malloc(N*sizeof(float));
	ones = (float*)malloc(n*sizeof(float));
	y = (float*)malloc(m*sizeof(float));

	for (int i = 0; i < N; i++) {
		if (i < n) {
			x[i] = (i%n)/100.0;
		} else {
			x[i] = (i%n)/100.0;
		}
	}

	cudaMalloc(&d_x, N*sizeof(float));
	cudaMalloc(&d_y, m * sizeof(float));

	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);

#if 0
	cudaMalloc(&d_ythreads, m*NUMBER_OF_THREADS*sizeof(float)); 
	
	cudaEventRecord(start);	

	addRow2Threads<<<m, NUMBER_OF_THREADS>>>( n, d_x, d_ythreads);
	//addThreads<<<500, 50>>>(n, d_ythreads, d_y);
Threads
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
#endif

#if 1
	cudaEventRecord(start);
	reduceColumn<<<m, NUMBER_OF_THREADS>>>(n, d_x, d_y);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
#endif
#if 0
	
	cudaMalloc(&d_ones, n*sizeof(float)); 
		for (int i = 0; i < n; i++) {
		ones[i] = 1.0f;
	}
	cudaMemcpy(d_ones, ones, n*sizeof(float), cudaMemcpyHostToDevice);
	// Create a handle for CUBLAS
	cublasHandle_t handle;
    cublasStatus_t stat;
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
    }    
	float alpha = 1;
	float beta = 0;
	cudaEventRecord(start);
	cublasSgemv(handle, CUBLAS_OP_T, n, m, &alpha, (const float*) d_x, n, (const float*) d_ones, 1, &beta, d_y, 1);
	cudaDeviceSynchronize();
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&milliseconds, start, stop);
	cublasDestroy(handle);
#endif

	cudaMemcpy(y, d_y, m*sizeof(float), cudaMemcpyDeviceToHost);
	ROS_INFO_STREAM("y: "<< y[0]);
	ROS_INFO_STREAM(" and Last value: "<< y[m-1] << " In time "<< milliseconds<< " with start ip "<< x[100]<< " end ip "<< x[31999]);

	cudaFree(d_x);
	cudaFree(d_y);
	free(x);
	free(y);
}

}

