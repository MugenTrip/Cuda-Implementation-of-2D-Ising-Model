#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__device__ void set_shared(int *buff, int* G,int off1 , int off2,int n)
{
	int m = blockIdx.x+off1*gridDim.x;
	int l = threadIdx.x+off2*blockDim.x;
	int maxx = blockDim.x-1;

	if(m<n && l<n)
	{	
		//If we reach the last element check if n is less than the number of threads
		//or if it's the last element of the current row/block
		if(l==n-1){
			if(blockDim.x > n)
				maxx = n-1;
			else if(n/blockDim.x==off2)
				maxx = (n-1)%blockDim.x;
		}	

		//The first element of each block will write the left corner 
		//and the last element the right one 
		if((threadIdx.x ==0 || threadIdx.x==maxx) && maxx!=0)
		{
			int ad;
			//Check if it's the first or the last thread of the block
			if(threadIdx.x==0)
				ad = -2;
			else
				ad = 0;

			for (int i = m-2; i <= m+2 ; i++)
			{
				for (int j = l+ad; j <= l+ad+2; j++)
				{
					int h1 = i - m;
					int h2 = j - l;
					int b_ind_x = 2+h1;
					int b_ind_y = threadIdx.x+2+h2;
					int g_ind_x = (n+i)%n;
					int g_ind_y = (n+j)%n;
					buff[b_ind_x*(blockDim.x+4)+b_ind_y] = G[g_ind_x*n+g_ind_y];
				}
			}		
		}
		//Special case for maxx==0:it means the first element is the last one too
		//it is necessary to write both sides in share memory
		else if(threadIdx.x==maxx && maxx==0)
		{
			for (int i = m-2; i <= m+2 ; i++)
			{
				for (int j = l-2; j <= l+2; j++)
				{
					int h1 = i - m;
					int h2 = j - l;
					int b_ind_x = 2+h1;
					int b_ind_y = threadIdx.x+2+h2;
					int g_ind_x = (n+i)%n;
					int g_ind_y = (n+j)%n;
					buff[b_ind_x*(blockDim.x+4)+b_ind_y] = G[g_ind_x*n+g_ind_y];
				}
			}		
		}
		//write only the values above and bellow you
		else
		{
			for (int i = m-2; i <= m+2 ; i++)
			{
				int h1 = i - m;
				int b_ind_x = 2+h1;
				int b_ind_y = threadIdx.x+2;
				int g_ind_x = (n+i)%n;
				int g_ind_y = (n+l)%n;
				buff[b_ind_x*(blockDim.x+4)+b_ind_y] = G[g_ind_x*n+g_ind_y];
			}

		}
	}
}

__global__ void gpu_update_sign(int *G, double *w , int k , int n ,int *temp, int *flag,int it_b ,int it_t)
{
	
	int buf=0;

	 __shared__ int buff[5140];
	
	for (int off1 = 0; off1 < it_b; off1++)
	{
		for(int off2 = 0; off2<it_t;off2++){
			//set share memory in every iteration
			set_shared(buff,  G, off1 ,  off2, n);

			int result;
			double sum = 0.0;

			//Find the indexes
			int x = blockIdx.x+off1*gridDim.x;
			int y = threadIdx.x+off2*blockDim.x;
			//Sync thread to be sure the share memory is ok
			__syncthreads();
		
			if(x<n && y<n){
				//Calculate the result
				for (int i = 0; i < k; i++){
					for (int j = 0; j < k; j++){
						sum += ((double)buff[i*(blockDim.x+4)+(threadIdx.x+j)])*w[i*k+j];	
					}
				}
				//Evaluate it
				if ( sum > 1e-6){
					result = 1; 
					if (result != buff[2*(blockDim.x+4)+threadIdx.x+2])
						buf++;
				}
				else if( sum < -(1e-6)){
					result = -1;
					if (result != buff[2*(blockDim.x+4)+threadIdx.x+2])
						buf++;
				}
				else
					result = buff[2*(blockDim.x+4)+threadIdx.x+2];
				//write to final array
				temp[x*n+y] =result;
			}
			//For stability of share memory setting
			__syncthreads();
		}
		//For stability of share memory setting
		__syncthreads();
	}
	*flag+=buf;
}

void ising_parallel(int *G, double *w, int k, int n)
{

	int number_of_threads = 1024;
	int number_of_block = 256;

	double it_b = (double) n / (double) number_of_block;
	it_b = (ceil(it_b));
	double it_t = (double) n / (double) number_of_threads;
	it_t = (ceil(it_t));

	int *G_cuda ,*temp;

	double *w_cuda;
	cudaMalloc((void **) &w_cuda,sizeof(double)*5*5);
	cudaMalloc((void **) &temp , sizeof(int)*n*n);
	cudaMalloc((void **) &G_cuda,sizeof(int)*n*n);

	int *flag_cuda , *flag;
	cudaMalloc((void **) &flag_cuda , sizeof(int));
	flag = (int *) malloc(sizeof(int));
	
	cudaMemcpyAsync(G_cuda,G,sizeof(int)*n*n,cudaMemcpyHostToDevice,NULL);
	cudaMemcpyAsync(w_cuda,w,sizeof(double)*5*5,cudaMemcpyHostToDevice,NULL);
	cudaMemcpyAsync(flag_cuda,flag,sizeof(int),cudaMemcpyHostToDevice,NULL);	


	for (int i = 0; i < k; i++)
	{
		gpu_update_sign<<<number_of_block,number_of_threads>>>(G_cuda,w_cuda,5,n,temp,flag_cuda,it_b,it_t);
		checkCuda( cudaGetLastError() );
		checkCuda( cudaMemcpy(G_cuda,temp,sizeof(int)*n*n,cudaMemcpyDeviceToDevice) );
		checkCuda( cudaMemcpy(flag,flag_cuda,sizeof(int),cudaMemcpyDeviceToHost));
		if(flag==0){
			break;
		}	
	}

	checkCuda(cudaMemcpy(G,G_cuda,sizeof(int)*n*n,cudaMemcpyDeviceToHost) );

	cudaFree(G_cuda);
	cudaFree(w_cuda);
	cudaFree(temp);
	free(flag);
}