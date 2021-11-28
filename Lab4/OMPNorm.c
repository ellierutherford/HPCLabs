#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <sys/time.h>
#include <math.h>
#include <stdbool.h>
#define MAXTHRDS 124

void PrintMatrix(double *matrixToPrint);
void InitializeMatrix(int seed, double *matrix);
double CalculateMatrixNorm(double* result);
void MultiplyMatrices(double *mat1, double *mat2, double *result);

int n;
int num_of_thrds;
int chunk_size;

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

int main(int argc,char* argv[])
{
    srand(time(NULL));
    double *mat1, *mat2, *result;

    struct timeval tv1, tv2;
    struct timezone tz;

    // blas also uses openmp in matrix multiplication calculation
    // without setting the number of threads blas can use to 1, it spits out loads of warnings
    openblas_set_num_threads(1);

    int i;
    num_of_thrds = omp_get_num_procs();
    n = atoi(argv[1]);
    omp_set_num_threads(num_of_thrds);
    if(num_of_thrds > n){
        chunk_size = n;
    }
    else if(num_of_thrds <= n){
        chunk_size = n/num_of_thrds;
    }

    mat1 = malloc(n*n*sizeof(double));
    mat2 = malloc(n*n*sizeof(double));
    result = malloc(n*n*sizeof(double));

    InitializeMatrix(1, mat1);
    InitializeMatrix(1, mat2);
    InitializeMatrix(0, result);

    // start timer
    gettimeofday(&tv1, &tz);

    // step 1: multiply matrices
    MultiplyMatrices(mat1,mat2,result);

    // step 2: calculate norm of resulting matrix
    double global_norm = CalculateMatrixNorm(result);

    // finish timer
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("Elapsed time is %f\n", elapsed);

    free(mat1);
    free(mat2);
    free(result);
}

double CalculateMatrixNorm(double* result){
    double my_sum=0;
    double global_norm=0;
    double my_norm=0;
    #pragma omp parallel for schedule(static,chunk_size) private(my_sum,my_norm) shared(global_norm)
    for(int i=0; i<n; i++){
        // inner loop takes care of getting each value in a given column
        for(int j=0;j<n;j++){
            // add the absolute value of each cell in column to total sum for column
            my_sum += fabs(result[i+j*n]);
        }
        // once you have the sum for the column, compare it to the current norm of the columns the thread owns
        // if it's greater than the current norm, update the norm of the thread and compare it to the global_norm
        // if the global norm is less than the thread's norm, update the global norm accordingly
        if(my_norm<my_sum){
            my_norm = my_sum;
            #pragma omp critical
	    if(global_norm < my_norm){
                global_norm = my_norm;
            }
        }
        // reset the sum in between columns
        my_sum = 0;
    }
    return global_norm;
}

void MultiplyMatrices(double* mat1, double* mat2, double* result){
    #pragma omp parallel for schedule(static,chunk_size)
    for(int i=0; i<num_of_thrds; i++) {
        int slice_size = n/num_of_thrds;
	int stride = (i==num_of_thrds-1) ? n-(num_of_thrds-1)*slice_size: slice_size;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, stride, n, 1, mat1, n, mat2 + i*slice_size, n, 1, result + i*slice_size, n);
    }
}

