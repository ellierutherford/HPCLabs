#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <cblas.h>
#include <math.h>
#include <stdbool.h>
#include <string.h>

#define MAXTHRDS 124

void PrintMatrix(double *matrixToPrint);
void InitializeMatrix(int seed, double *matrix);
double CalculateMatrixNorm(double *result);
void MultiplyMatrices(double *mat1, double *mat2, double *result);
bool TestNorm(double *mat1, double *mat2, double *actualResult, double *expectedResult, double expectedNorm, int num_of_thrds, int n);
void test();
bool VerifyArraysEqual(double *actual, double *expected);

int n, num_of_thrds;

typedef struct {
    double *mat1;
    double *slice;
    int slice_size;
    double *result;
}matrix_multiply_t;

typedef struct {
    double my_sum;
    double *global_norm;
    pthread_mutex_t *mutex;
    double *result;
    int num_of_cols;
} matrix_norm_t;

void *matrix_norm(void *arg) {
    matrix_norm_t *norm_data;
    int i,j;
    norm_data = arg;
    int num_of_cols = norm_data->num_of_cols;
    // outer loop ensures we only get the sum of the columns assigned to this thread
    for(i=0; i<num_of_cols; i++){
        // inner loop takes care of getting each value in a given column
        for(j=0;j<n;j++){
            // add the absolute value of each cell in column to total sum for column
            norm_data->my_sum += fabs(norm_data->result[i+j*n]);
        }
        // once you have the sum for the column, compare it to the 'global' norm for the matrix
        // if the global norm is less than the sum of this column, update the norm to be this column's sum
        pthread_mutex_lock(norm_data->mutex);
        if(*(norm_data->global_norm) < norm_data->my_sum){
            *(norm_data->global_norm) = norm_data->my_sum;
        }
        // reset the sum in between columns
        norm_data->my_sum = 0;
        pthread_mutex_unlock(norm_data->mutex);
    }
    pthread_exit(NULL);
}

void *matrix_multiply(void *arg) {
    matrix_multiply_t *slice_data;
    slice_data = arg;
    int slice_size = slice_data->slice_size;
    // multiply matrix 1 (left matrix) with thread's slice, storing result in corresponding slice of result matrix
    // the slice variable points to the address in memory where the slice begins, the slice_size specifies the number of columns in the slice
    // and the major stride is set to n i.e. the number of spaces in memory to skip to get to the next element in the column
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, slice_size, n, 1, slice_data->mat1, n, slice_data->slice, n, 1, slice_data->result, n);
    pthread_exit(NULL);
}

void MultiplyMatrices(double *mat1, double *mat2, double *result){
    matrix_multiply_t *thrd_mat_mul_data;
    pthread_t *working_thread;
    void *status;
    working_thread = malloc(num_of_thrds*sizeof(pthread_t));
    thrd_mat_mul_data = malloc(num_of_thrds*sizeof(matrix_multiply_t));
    int slice_size = n/num_of_thrds;
    int i;
    // use pthreads to perform the matrix multiplication
    for(i=0; i<num_of_thrds; i++) {
        thrd_mat_mul_data[i].mat1 = mat1;
	int stride = (i==num_of_thrds-1) ? n-(num_of_thrds-1)*slice_size: slice_size;
        thrd_mat_mul_data[i].result = result + i*slice_size;
        thrd_mat_mul_data[i].slice = mat2 + i*slice_size;
        thrd_mat_mul_data[i].slice_size = stride;
        pthread_create(&working_thread[i], NULL, matrix_multiply,(void*)&thrd_mat_mul_data[i]);
    }

    for(i=0; i<num_of_thrds; i++){
         pthread_join(working_thread[i], &status);
    }

    free(working_thread);
    free(thrd_mat_mul_data);
}

double CalculateMatrixNorm(double *result){
    double norm = 0;
    void *status;
    pthread_t *norm_threads;
    matrix_norm_t *thrd_norm_data;
    pthread_mutex_t *mutex_norm;
    norm_threads = malloc(num_of_thrds*sizeof(pthread_t));
    thrd_norm_data = malloc(num_of_thrds*sizeof(matrix_norm_t));
    mutex_norm = malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mutex_norm, NULL);
    // use pthreads to calculate the norm of the resulting matrix
    int num_of_cols = n/num_of_thrds;
    int i;
    for(i=0;i<num_of_thrds; i++){
        thrd_norm_data[i].result = result + i*num_of_cols;
        thrd_norm_data[i].my_sum = 0;
        thrd_norm_data[i].global_norm = &norm;
        thrd_norm_data[i].mutex = mutex_norm;
        thrd_norm_data[i].num_of_cols = (i==num_of_thrds-1)? n-(num_of_thrds-1)*num_of_cols: num_of_cols;
        pthread_create(&norm_threads[i], NULL, matrix_norm, (void*)&thrd_norm_data[i]);
    }
    for(i=0;i<num_of_thrds;i++){
        pthread_join(norm_threads[i], &status);
    }

    free(norm_threads);
    free(thrd_norm_data);
    pthread_mutex_destroy(mutex_norm);
    free(mutex_norm);

    return norm;
}

int main(int argc,char* argv[])
{
    srand(time(NULL));
    struct timeval tv1, tv2;
    struct timezone tz;

    // COMMENTED OUT USER INPUT - RUNNING EXPERIMENTS EASIER WITH CMD LINE ARGS
    /*char c;
    printf("Press t to run tests or any other key to continue to main program: ");
    scanf("%c",&c);
    if(c=='t'){
        test();
        return 0;
    }
    printf("Number of processors = ");
    if(scanf("%d", &num_of_thrds) < 1 || num_of_thrds > MAXTHRDS) {
        printf("Check input for number of processors. Bye.\n");
        return -1;
    }
    printf("Matrix size = ");
    if(scanf("%d", &n)<1) {
        printf("Check input for matrix size. Bye.\n");
        return -1;
    }*/

    if(argc==2 & *argv[1]=='t'){
        test();
        return 0;
    }

    num_of_thrds = atoi(argv[1]);
    if(num_of_thrds > MAXTHRDS){
        printf("too many threads! Exiting...");
        return -1;
    }
    n = atoi(argv[2]);

    double *mat1, *mat2, *result;

    mat1 = malloc(n*n*sizeof(double));
    mat2 = malloc(n*n*sizeof(double));
    result = malloc(n*n*sizeof(double));

    InitializeMatrix(1, mat1);
    InitializeMatrix(1, mat2);
    InitializeMatrix(0, result);

    // start timer
    gettimeofday(&tv1, &tz);

    // step 1: get result matrix
    MultiplyMatrices(mat1,mat2,result);
    // step 2: calculate norm of result matrix
    double norm = CalculateMatrixNorm(result);
    //printf("\nMatrix norm is %f\n", norm);

    // finish timer
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
    printf("Elapsed time is %f\n", elapsed);

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

void PrintMatrix(double *matrixToPrint){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%lf ", matrixToPrint[i*n + j]);
        }
        printf("\n");
    }
}

void InitializeMatrix(int seed, double *matrix){
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            matrix[i*n + j] = ((rand()%10+1) * seed);
        }
    }
}
