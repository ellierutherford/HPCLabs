
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

int main(int argc, char *argv[])
{
    /*if(argc != 2)
    {
       printf("Plese, use: %s N, where N is matrix dimension", argv[0]);
       exit(EXIT_FAILURE);
    }*/

    struct timeval tv1, tv2;
    struct timezone tz;

    //double *A, *B, *C;
    double *C;
    int N,i,j,k;

    N = 5;//atoi(argv[1]);

    //A = malloc(N*N*sizeof(double));
    //B = malloc(N*N*sizeof(double));
    C = malloc(N*N*sizeof(double));

    /*for(i=0; i<N*N; i++)
    {
        A[i] = 1.;
        B[i] = 3.;
        C[i] = 0.;
    }*/

    double A[] = {7,4,2,9,4,0,1,6,5,5,9,3,9,3,0,4,0,5,3,5,7,8,0,2,8};
    double B[] = {8,1,8,0,3,5,4,0,1,5,5,6,1,0,4,0,4,5,6,3,4,7,0,0,6};
    
    gettimeofday(&tv1, &tz);

    for(i=0; i<N; i++){
       for(j=0; j<N; j++){
          for(k=0; k<N; k++){
              C[i*N + j] += A[i*N + k] * B[k + j*N];
          }
       }
      printf("\n%f ", C[i*N + j]);
    }
    
    gettimeofday(&tv2, &tz);
    double elapsed = (double) (tv2.tv_sec-tv1.tv_sec) + (double) (tv2.tv_usec-tv1.tv_usec) * 1.e-6;


    for(i=0; i<N; i++){
       for(j=0; j<N; j++){
           printf("%f\t", C[i*N + j]);
       }
           printf("\n");
    }   

    printf("Elapsed time is %f\n", elapsed);
    exit(0);
}

