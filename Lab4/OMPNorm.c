#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <cblas.h>
#include <time.h>
#include <math.h>
#define MAXTHRDS 124

void PrintMatrix(double *matrixToPrint);
void InitializeMatrix(int seed, double *matrix);
bool TestNorm(double *mat1, double *mat2, double *actualResult, double *expectedResult, double expectedNorm, int num_of_thrds, int n);
void test();
bool VerifyArraysEqual(double *actual, double *expected);

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
    
    if(argc==2 & *argv[1]=='t'){
        test();
        return 0;
    }
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

    double my_sum = 0;
    double global_norm = 0;

    #pragma omp parallel for schedule(dynamic,num_of_thrds) private(my_sum) shared(global_norm)
    for(int i=0; i<n; i++){
        // inner loop takes care of getting each value in a given column
        for(int j=0;j<n;j++){
            // add the absolute value of each cell in column to total sum for column
            my_sum += fabs(result[i+j*n]);
        }
        //printf("my sum is %lf and %d\n", my_sum, omp_get_thread_num());
        // once you have the sum for the column, compare it to the 'global' norm for the matrix
        // if the global norm is less than the sum of this column, update the norm to be this column's sum
        #pragma omp critical
        if(global_norm < my_sum){
            global_norm = my_sum;
	    //printf("global norm is now %lf and thread id %d\n", global_norm, omp_get_thread_num());
        }
        // reset the sum in between columns
        my_sum = 0;
    }

    printf("norm is %lf\n",global_norm);

    free(mat1);
    free(mat2);
    free(result);
}

void test(){
    printf("Running tests\n");
    printf("\nTest 1\n");
    double mat1[] = {136,158,112,122,100,123,96,32,160,102,175,27,163,93,104,164};
    double mat2[] = {6,1,8,8,6,9,9,4,9,1,2,6,7,2,5,7};
    double expected_result[] = {3626, 1914, 3344, 3246, 2426, 1367, 2259, 2092, 3336, 1307, 2683, 2927, 3620, 1432, 3169, 3448};
    double *actual_result;
    int n = 4;
    int num_of_threads = 2;
    bool test_pass = TestNorm(mat1,mat2,actual_result,expected_result,13008,num_of_threads,n);

    printf("\nTest 2\n");
    num_of_threads = 3;
    test_pass = TestNorm(mat1,mat2,actual_result,expected_result,13008,num_of_threads,n);

    printf("\nTest 3\n");
    double mat3[] = {-2,0,4,4,-1,4,-2,0,3,4,2,1,5,2,5,0,-2,-1,4,-2,0,0,1,-2,5,1,5,3,3,4,4,1,3,4,4,-1};
    double mat4[] = {-1,2,5,0,5,1,4,0,-3,5,4,0,-1,4,2,5,5,3,3,3,1,1,-3,1,1,0,0,4,5,1,-3,1,5,5,3,5};
    double expected_result2[] = {-3, 28, 22, 40, 5, 33, 10, 21, 5, 32, 6, 18, -1, 29, 24, 22, 45, 13, -5, 6, 16, -16, 11, -5, -6, 43, 55, 65, 72, 46, 16, 31, 22, 35, 44, 16};
    num_of_threads = 5;
    n = 6;
    test_pass = TestNorm(mat3,mat4,actual_result,expected_result2,210,num_of_threads,n);

    printf("\nTest 4\n");
    num_of_threads = 1;
    test_pass = TestNorm(mat3,mat4,actual_result,expected_result2,210,num_of_threads,n);
}

bool TestNorm(double *mat1, double *mat2, double *actual_result, double *expected_result, double expected_norm, int num_of_threads, int matrixSize){
    num_of_thrds = num_of_threads;
    n = matrixSize;
    actual_result = malloc(n*n*sizeof(double));
    InitializeMatrix(0,actual_result);
    MultiplyMatrices(mat1,mat2,actual_result);
    double norm = CalculateMatrixNorm(actual_result);
    bool test_passed = VerifyArraysEqual(actual_result,expected_result) && norm == expected_norm;
    free(actual_result);
    printf("Matrix norm for n=%d, num_of_thrds=%d %s\n", n, num_of_threads, test_passed ? "passed" : "failed");
    return test_passed;
}

bool VerifyArraysEqual(double *actual, double *expected){
    for(int i=0;i<n*n;i++){
        if(actual[i]!=expected[i]){
	    return false;
	}
    }
    return true;
}
