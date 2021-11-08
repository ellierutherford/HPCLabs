#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/time.h>
#include <cblas.h>
#include <math.h>

#define MAXTHRDS 124

void PrintMatrix(double *matrixToPrint);
void InitializeMatrix(int seed, double *matrix);
int n;

typedef struct {
    double *mat1;
    double *slice;
    int slice_size;
    double *result;
} matrix_multiply_t;

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
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, slice_size, n, 1, slice_data->mat1, n, slice_data->slice, n, 1, slice_data->result, n);
    pthread_exit(NULL);
}
int main()
{
    srand(time(NULL));
    struct timeval tv1, tv2;
    struct timezone tz;
    double *mat1, *mat2, *result;
    double norm = 0;
         //double mat1[] = {136,158,112,122,100,123,96,32,160,102,175,27,163,93,104,164};
    	 //double mat2[] = {6,1,8,8,6,9,9,4,9,1,2,6,7,2,5,7};
	 //double mat1[] = {7,1,8,7,9,0,9,0,1,8,0,8,5,3,0,8};
 	 //double mat2[] = {1,2,5,4,1,9,7,3,2,9,3,8,9,2,8,3};
    pthread_t *working_thread, *norm_threads;
    matrix_multiply_t *thrd_mat_mul_data;
    matrix_norm_t *thrd_norm_data;
    pthread_mutex_t *mutex_norm;

    void *status;
    int num_of_thrds;
    int slice_size;
    int i;
    printf("Number of processors = ");
    if(scanf("%d", &num_of_thrds) < 1 || num_of_thrds > MAXTHRDS) {
        printf("Check input for number of processors. Bye.\n");
        return -1;
    }
    printf("Matrix size = ");
    if(scanf("%d", &n)<1) {
        printf("Check input for matrix size. Bye.\n");
        return -1;
    }

    slice_size = n/num_of_thrds;
    mat1 = malloc(n*n*sizeof(double));
    mat2 = malloc(n*n*sizeof(double));
    result = malloc(n*n*sizeof(double));

    InitializeMatrix(1, mat1);
	 //printf("\nprinting %dx%d matrix 1\n", n, n);
	 //PrintMatrix(mat1);
    InitializeMatrix(1, mat2);
	 //printf("\nprinting %dx%d matrix 2\n", n, n);
         //PrintMatrix(mat2);
         //printf("try initialize results matrix");
    InitializeMatrix(0, result);

    working_thread = malloc(num_of_thrds*sizeof(pthread_t));
    thrd_mat_mul_data = malloc(num_of_thrds*sizeof(matrix_multiply_t));
    norm_threads = malloc(num_of_thrds*sizeof(pthread_t));
    thrd_norm_data = malloc(num_of_thrds*sizeof(matrix_norm_t));
    mutex_norm = malloc(sizeof(pthread_mutex_t));
    pthread_mutex_init(mutex_norm, NULL);
    gettimeofday(&tv1, &tz);
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

    int num_of_cols = n/num_of_thrds;
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
    printf("\nmatrix norm is %f\n", norm);
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;
         //PrintMatrix(result);
    printf("Elapsed time is %f\n", elapsed);
    free(mat1);
    free(mat2);
    free(result);
    free(working_thread);
    free(thrd_mat_mul_data);
    free(norm_threads);
    free(thrd_norm_data);
    pthread_mutex_destroy(mutex_norm);
    free(mutex_norm);
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
