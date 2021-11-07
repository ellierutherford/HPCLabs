#include <stdio.h>
#include <stdlib.h>
#include "MatrixMultiplication.h"
#include <sys/time.h>
#include<string.h>
#include <unistd.h>

// number of rows and columns in nxn matrices
int n;

int main(int argc,char* argv[]){
    n = atoi(argv[1]);
    struct timeval tv1, tv2;
    struct timezone tz;

    double *mat1;
    double *mat2;
    double *result;
    mat1 = malloc(n*n*sizeof(double));
    mat2 = malloc(n*n*sizeof(double));
    result = malloc(n*n*sizeof(double));

    // initialize a and b with non zero values and c with all zero values
    InitializeMatrix(1,mat1);
    sleep(1); // sleep so we get different random numbers in each matrix
    InitializeMatrix(1,mat2);
    InitializeMatrix(0,result);

    // uncomment to print generated matrices for debugging purposes
    /*printf("mat1:\n");
    PrintMatrix(mat1);
    printf("mat2:\n");
    PrintMatrix(mat2);
    printf("result:\n");*/

    gettimeofday(&tv1, &tz);
    
    // use input to determine which matrix multiplication method to use
    if(strcmp(argv[2],"b")==0){
	printf("Doing blocked ijk\n");
	int b = atoi(argv[3]);
	BlockedMultiply(mat1, mat2, result, b);
    }
    else if(strcmp(argv[2],"nb")==0){
        printf("Doing non blocked ijk\n");
	NonBlockedMultiply(mat1,mat2,result);
    }
    else if(strcmp(argv[2],"kij")==0){
        printf("Doing non blocked kij\n");
        NonBlockedKij(mat1,mat2,result);
    }
    else if(strcmp(argv[2],"blas")==0){
        printf("Doing blas\n");
        MultiplyBlas(mat1,mat2,result);
    }
    else if(strcmp(argv[2],"bblas")==0){
        printf("Doing blocked kij blas\n");
        int b = atoi(argv[3]);
        BlockedKijBlas(mat1,mat2,result,b);
    }
    else if(strcmp(argv[2],"bkij")==0){
        printf("Doing blocked kij\n");
        int b = atoi(argv[3]);
 	BlockedKij(mat1,mat2,result,b);
    }
    
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("\nElasped time is %lf\n", elapsed);
    
    // free up the memory allocated for all 3 matrices
    free(mat1);
    free(mat2);
    free(result);

}
