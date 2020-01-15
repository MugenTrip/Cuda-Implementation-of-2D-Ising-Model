#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include "ising.h"

int update_sign(int *G, double *w ,int *neighbors , int k , int n)
{
	int counter=0;
	int *temp = (int *) malloc(sizeof(int)*n*n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			
			double sum = 0.0;
			for (int l = 0; l < k; l++)
			{
				for (int m = 0; m < k; m++)
				{
					int index = i*n*k*k+j*k*k+l*k+m;
					sum += ((double)G[neighbors[index]])*w[l*k+m];	
				}
			}

			if ( sum > 1e-6){
				temp[i*n+j] = 1;
				if (temp[i*n+j] != G[neighbors[i*n*k*k+j*k*k+12]])
					counter++;
			}
			else if( sum < -(1e-6)){
				temp[i*n+j] = -1;
				if (temp[i*n+j] != G[neighbors[i*n*k*k+j*k*k+12]])
					counter++;
			}
			else{
				temp[i*n+j] = G[neighbors[i*n*k*k+j*k*k+12]];
			}
		}
	}
	memcpy(G, temp, n*n*sizeof(int));
	free(temp);
	return counter;
}

void get_neighbors(int *neighbors, int n , int k)
{

	for (int m = 0; m < n; m++)
	{
		for (int l = 0; l < n; l++)
		{
			int counter_i =0;
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

void ising(int *G, double *w, int k, int n)
{
	//Calculate the 5x5 neighborhood of every point
	int *neighbors = (int *) malloc(sizeof(int)*n*n*25);
	get_neighbors(neighbors , n , 5);
	
	int counter =0;			
	for (int i = 0; i < k; i++)
	{
		int flag;
		flag = update_sign(G,w,neighbors,5,n);
		counter++;
		if (flag==0){
			printf("breakpoint %d \n",counter );
			break;
		}
	}
}