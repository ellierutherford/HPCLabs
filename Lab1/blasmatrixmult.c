#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
       printf("Plese, use: %s N, where N is matrix dimension", argv[0]);
       exit(EXIT_FAILURE);
    }

    struct timeval tv1, tv2;
    struct timezone tz;

    double *A, *B, *C;
    int N,i,j,k;

    N = atoi(argv[1]);

    A = malloc(N*N*sizeof(double));
    B = malloc(N*N*sizeof(double));
    C = malloc(N*N*sizeof(double));

    for(i=0; i<N*N; i++)
    {
        A[i] = 1.;
        B[i] = 3.;
        C[i] = 0.;
    }

    gettimeofday(&tv1, &tz);

    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1.0, A, N, B, N, 0.0, C, N);

    gettimeofday(&tv2, &tz);
    
    for(i=0; i<3; i++){
       for(j=0; j<3; j++){
           printf("%f\t", C[i*N + j]);
       }
           printf("\n");
    }   
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;

    printf("Elapsed time is %f\n", elapsed);
    exit(0);
}

