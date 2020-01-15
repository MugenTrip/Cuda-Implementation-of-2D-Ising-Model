#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include "ising.h"

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__ void gpu_get_neighors(int *neighbors, int n , int k)
{
	for (int off1 = 0; off1 < n/gridDim.x+1 ; off1++)
	{
		for(int off2 = 0; off2 < n/blockDim.x+1 ;off2++){
			
			int m = blockIdx.x+off1*gridDim.x;
			int l = threadIdx.x+off2*blockDim.x;

			int counter_i =0;
			if(m<n && l<n){
				for (int i = m-(k/2); i <= m+(k/2); i++)
				{
					int counter_j=0;
					for (int j = l-(k/2); j <= l+(k/2); j++)
					{
						int index , index_i , index_j;
						index = m*n*k*k + l*k*k + counter_i*k +counter_j;
						index_i =(n+i)%n;
						index_j=(n+j)%n;
						neighbors[index] = index_i*n+index_j;
						counter_j++; 
					}
					counter_i++;
				}
			}
		}
	}
}

__global__ void gpu_update_sign(int *G, double *w ,int *neighbors , int k , int n ,int *temp, int *flag,int it_b ,int it_t)
{
	int result;
	double sum = 0.0;
	int buf=0;
	//Find the indexes
	int x = blockIdx.x+it_b*gridDim.x;
	int y = threadIdx.x+it_t*blockDim.x;

	if (blockIdx.x+it_b*gridDim.x<n && threadIdx.x+it_t*blockDim.x<n)
	{
		//Calculate result	
		for (int i = 0; i < k; i++){
			for (int j = 0; j < k; j++){
				sum += ((double)G[neighbors[x*n*k*k+y*k*k+i*k+j]])*w[i*k+j];	
			}
		}
		//Evaluate and write back
		if ( sum > 1e-6){
			result = 1; 
			if (result != G[neighbors[x*n*k*k+y*k*k+12]])
				buf++;
		}
		else if( sum < -(1e-6)){
			result = -1;
			if (result != G[neighbors[x*n*k*k+y*k*k+12]])
				buf++;
		}
		else
			result = G[neighbors[x*n*k*k+y*k*k+12]];
	
		*flag+=buf;
		temp[x*n+y] =result;	
	}
}

void ising_parallel(int *G, double *w, int k, int n)
{	
	//Setting number of of threads and blocks
	int number_of_threads = 1024;
	int number_of_block = 256;
	
	//Calculation the number of the itterarions that they will be needed
	double it_b = (double) n / (double) number_of_block;
	it_b = (ceil(it_b));
	double it_t = (double) n / (double) number_of_threads;
	it_t = (ceil(it_t));
	
	//Initiate the parameters in the memory of the device 
	int *G_cuda , *neighbors_cuda, *temp;
	cudaMalloc((void **) &G_cuda,sizeof(int)*n*n);
	cudaMalloc((void **) &neighbors_cuda,sizeof(int)*n*n*25);
	cudaMalloc((void **) &temp , sizeof(int)*n*n);
	
	double *w_cuda;
	cudaMalloc((void **) &w_cuda,sizeof(double)*5*5);
	
	int *flag_cuda , *flag;
	flag = (int *) malloc(sizeof(int));
	cudaMalloc((void **) &flag_cuda , sizeof(int));
	
	cudaMemcpyAsync(G_cuda,G,sizeof(int)*n*n,cudaMemcpyHostToDevice,NULL);
	cudaMemcpyAsync(w_cuda,w,sizeof(double)*5*5,cudaMemcpyHostToDevice,NULL);
	
	//Calculation the indexes of the neighbors in parallel
	gpu_get_neighors<<<number_of_block,number_of_threads>>>(neighbors_cuda , n , 5);
	checkCuda( cudaGetLastError());
	cudaDeviceSynchronize();

	//Starting the k times itteration
	int counter=0;
	for (int i = 0; i < k; i++)
	{
		//Calling it_b*it_t times the kernel function to make the complete update
		int buff=0;
		for(int j=0; j<(int)it_b;j++)
		{
			for (int l = 0; l< (int)it_t; l++)
			{
				gpu_update_sign<<<number_of_block,number_of_threads>>>(G_cuda,w_cuda,neighbors_cuda,5,n,temp,flag_cuda,j,l);
				checkCuda( cudaGetLastError());
				checkCuda( cudaMemcpy(flag,flag_cuda,sizeof(int),cudaMemcpyDeviceToHost) );
				buff+=flag[0];
				counter++;
			}
		}
		//Setting data ready for the next itteration
		checkCuda( cudaMemcpy(G_cuda,temp,sizeof(int)*n*n,cudaMemcpyDeviceToDevice) );	

		//Check if there is no changes
		if(buff==0){
			printf("breakpoint %ds:\n",counter);
			break;
		}
	}

	//Writing the final result
	checkCuda(cudaMemcpy(G,G_cuda,sizeof(int)*n*n,cudaMemcpyDeviceToHost) );

	cudaFree(G_cuda);
	cudaFree(w_cuda);
	cudaFree(neighbors_cuda);
	cudaFree(temp);
}