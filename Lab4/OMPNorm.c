#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <time.h>
#include <math.h>
#define MAXTHRDS 124

void PrintMatrix(double *matrixToPrint);
void InitializeMatrix(int seed, double *matrix);

int n;

void InitializeMatrix(int seed, double *matrix){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            matrix[i*n + j] = ((rand()%10+1) * seed);
        }
    }
}

void PrintMatrix(double *matrixToPrint){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%lf ", matrixToPrint[i*n + j]);
        }
        printf("\n");
    }
}

int main()
{
    srand(time(NULL));
    double *mat1, *mat2, *result;

    // blas also uses openmp in matrix multiplication calculation
    // without setting the number of threads blas can use to 1, it spits out loads of warnings
    openblas_set_num_threads(1);

    int num_of_thrds, i;
    num_of_thrds = omp_get_num_procs();
    printf("num of threads: %d\n", num_of_thrds);
    omp_set_num_threads(num_of_thrds);
    printf("Matrix size = ");
    if(scanf("%d", &n)<1) {
        printf("Check input for matrix size. Bye.\n");
        return -1;
    }

    mat1 = malloc(n*n*sizeof(double));
    mat2 = malloc(n*n*sizeof(double));
    result = malloc(n*n*sizeof(double));

    InitializeMatrix(1, mat1);
    InitializeMatrix(1, mat2);
    InitializeMatrix(0, result);

    printf("Matrix1: \n");
    //PrintMatrix(mat1);
    printf("Matrix 2: \n");
    //PrintMatrix(mat2);

    #pragma omp parallel for schedule(dynamic,num_of_thrds)
    for(i=0; i<num_of_thrds; i++) {
        int slice_size = n/num_of_thrds;
	int stride = (i==num_of_thrds-1) ? n-(num_of_thrds-1)*slice_size: slice_size;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, stride, n, 1, mat1, n, mat2 + i*slice_size, n, 1, result + i*slice_size, n);
    }

    printf("Result: \n");
    //PrintMatrix(result);

    free(mat1);
    free(mat2);
    free(result);
}

