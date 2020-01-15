#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include "ising.h"

#define DOWN -1		//Spin down 
#define UP 1		//Spin up

int pop_rand();
void initiate_spin_grid(int *spins,int n);
int evaluation(int *Correct , int* query , int n);

int main()
{
	struct timeval startwtime,endwtime;
	double seq_time;


	int *G = (int *) malloc(sizeof(int)*1000*1000);
	int *G_cuda = (int *) malloc(sizeof(int)*1000*1000);

  	initiate_spin_grid(G,1000);
  	memcpy(G_cuda, G, 1000*1000*sizeof(int));

	double *w = (double *) malloc(sizeof(double)*5*5);
	w[0]=w[4]=w[20]=w[24]=0.004;
	w[1]=w[3]=w[5]=w[9]=w[15]=w[19]=w[21]=w[23]=0.016;
	w[2]=w[10]=w[14]=w[22]=0.026;
	w[6]=w[8]=w[16]=w[18]=0.071;
	w[7]=w[11]=w[13]=w[17]=0.117;
	w[12]=0.0;

	gettimeofday(&startwtime,NULL);

	ising(G,w,50,1000);

	gettimeofday(&endwtime,NULL);

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
    printf("Serial time = %f \n", seq_time);	


	gettimeofday(&startwtime,NULL);

	ising_parallel(G_cuda,w,50,1000);

	gettimeofday(&endwtime,NULL);

	seq_time = (double)((endwtime.tv_usec - startwtime.tv_usec)/1.0e6+endwtime.tv_sec - startwtime.tv_sec);
    printf("Parallel time = %f \n", seq_time);

	int value = evaluation(G , G_cuda , 1000);
	if (value==0)
		printf("Test Failed\n");
	else
		printf("Test Passed\n");
	return 0;
}

int evaluation(int *correct , int* query , int n)
{
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			if (query[i*n+j]!=correct[i*n+j])
			{
				printf("Error in position %d ",i*n+j);
				printf("%d != %d \n",query[i*n+j] , correct[i*n+j] );
				return 0;
			}
		}
	}
	return 1;
}

int pop_rand()
{
	int r = rand();
	if(r%2 == 0) return UP; 
	else	return DOWN; 
}

void initiate_spin_grid(int *spins,int n)
{
	for (int i = 0; i < n*n; i++)
	{
		spins[i] = pop_rand();
	}	 
}