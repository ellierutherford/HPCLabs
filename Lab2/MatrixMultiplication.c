#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include "MatrixMultiplication.h"
extern int n;

// Blocked ijk
void BlockedMultiply(double *mat1, double *mat2, double *result, int b){
    int i,j,k;
    int bi,bj,bk;
    int blockSize = b;
    for(bi=0; bi<n; bi+=blockSize){
        for(bj=0; bj<n; bj+=blockSize){
            for(bk=0; bk<n; bk+=blockSize){
		for(i=0; i<blockSize; i++){
                    for(j=0; j<blockSize; j++){
                        for(k=0; k<blockSize; k++){
                            result[(bi+i)*n + (bj+j)] += mat1[(bi+i)*n + (bk+k)] * mat2[(bk+k)*n + (bj+j)];
                        }
                    }
		}
            }
        }
    }
}

// Blocked Kij
void BlockedKij(double *mat1, double *mat2, double *result, int b){
    int i,j,k;
    int bi,bj,bk;
    int blockSize = b;
    for(bk=0; bk<n; bk+=blockSize){
        for(bi=0; bi<n; bi+=blockSize){
            for(bj=0; bj<n; bj+=blockSize){
		for(k=0; k<blockSize; k++){
                    for(i=0; i<blockSize; i++){
                        for(j=0; j<blockSize; j++){
                            result[(bi+i)*n + (bj+j)] += mat1[(bi+i)*n + (bk+k)] * mat2[(bk+k)*n + (bj+j)];
                        }
                    }
		}
            }
        }
    }
}

// Blocked Kij using Blas
void BlockedKijBlas(double *mat1, double *mat2, double *result, int b){
    int blockSize = b;
    int bk, bi, bj;

    for (bk = 0; bk < n ; bk += blockSize)
    {
        for (bi = 0; bi < n; bi += blockSize)
        {
            for (bj = 0; bj < n; bj += blockSize)
            {
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, blockSize, blockSize, blockSize, 1, mat1 + (bi*n) + bk, n, mat2 + bj + (bk*n), n, 1, result + (bi*n) + bj, n);
            }
        }
    }
}

// Straightforward non blocked ijk matrix multiply
void NonBlockedMultiply(double *mat1, double *mat2, double *result){
    int i,j,k;
    for (i=0;i<n;i++){
        for (j=0;j<n;j++){
            int sum = 0;
            for (k=0;k<n;k++) {
                sum += mat1[i*n + k] * mat2[j + k*n];
            }
            result[i*n + j] = sum;
        }
    }
}

// Non blocked Blas
void MultiplyBlas(double *mat1, double *mat2, double *result){
	cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0, mat1, n, mat2, n, 0.0, result, n);
}

// Straightforward non blocked kij matrix multiply
void NonBlockedKij(double *mat1, double *mat2, double *result){
    int i,j,k;
    for (k=0;k<n;k++){
        for (i=0;i<n;i++){
            int x = mat1[i*n + k];
            for (j=0;j<n;j++) {
                result[i*n + j] += x * mat2[j + k*n];
            }
        }
    }
}
