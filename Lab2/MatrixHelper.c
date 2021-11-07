#include <stdio.h>
#include <stdlib.h>
#include <time.h>

extern int n;

void PrintMatrix(double *matrixToPrint){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%lf ", matrixToPrint[i*n + j]);
        }
        printf("\n");
    }
}

// initialize a given matrix to random numbers between 1 and 10
// use the seed variable to optionally set all elements to 0 for initializing results matrix
void InitializeMatrix(int seed, double *matrix){
    srand(time(NULL));    
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            matrix[i*n + j] = ((rand()%10+1) * seed);
        }
    }
}

