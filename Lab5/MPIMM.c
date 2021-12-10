

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

void PrintMatrix(double *matrixToPrint);
void InitializeMatrix(int seed, double *matrix);

int n, p;
int main(int argc, char **argv) {
   int myn, myrank;
   double *a, *b, *c, *matA, *matB, *rowA, *colB, start, sum, *allC, sumdiag;
   int i, j, k;
   
   n = atoi(argv[1]);
   MPI_Comm rowComm, colComm;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD,&p);
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);

   int sqrtp = (int)sqrt(p);
   int numOfSquares = p;
   int squareSize = n/sqrtp;

   if(myrank==0){
       matA = malloc(n*n*sizeof(double));
       matB = malloc(n*n*sizeof(double));

       InitializeMatrix(1,matA);
       sleep(1); // sleep so we get different random numbers in each matrix
       InitializeMatrix(1,matB);
       printf("Printing mat A: \n");
       PrintMatrix(matA);
       printf("Printing mat B: \n");
       PrintMatrix(matB);
   }

   a = malloc(squareSize*squareSize*sizeof(double));
   b = malloc(squareSize*squareSize*sizeof(double));
   c = malloc(squareSize*squareSize*sizeof(double));
   rowA = malloc(squareSize*n*sizeof(double));
   colB = malloc(squareSize*n*sizeof(double));

   // send blocks of matrix A to all processors
   MPI_Request recvRequestA;
   int count=0;
   for(int i=0;i<squareSize;i++){
       MPI_Irecv(&a[i*squareSize],squareSize,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&recvRequestA);
       count++;
   }

   if(myrank==0){
       int currentProc=0;
       for(int i=0;i<sqrtp;i++){
           for(int j=0;j<sqrtp;j++){
               for(int k=0;k<squareSize;k++){
                   MPI_Send(&matA[(i*squareSize*n)+(j*squareSize)+(k*n)],squareSize,MPI_DOUBLE,currentProc,1,MPI_COMM_WORLD);
               }
               currentProc++;
           }
       }
   }

   MPI_Wait(&recvRequestA, MPI_STATUS_IGNORE);

   // send blocks of matrix B to all processors
   MPI_Request recvRequestB;
   for(int i=0;i<squareSize;i++){
       MPI_Irecv(&b[i*squareSize],squareSize,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&recvRequestB);
   }

   if(myrank==0){
       int currentProc=0;
       for(int i=0;i<sqrtp;i++){
           for(int j=0;j<sqrtp;j++){
               for(int k=0;k<squareSize;k++){
                   MPI_Send(&matB[(i*squareSize*n)+(j*squareSize)+(k*n)],squareSize,MPI_DOUBLE,currentProc,1,MPI_COMM_WORLD);
               }
               currentProc++;
           }
       }
   }

   MPI_Wait(&recvRequestB, MPI_STATUS_IGNORE);

   MPI_Barrier(MPI_COMM_WORLD);
   if(myrank==0)
     start = MPI_Wtime();

   MPI_Comm_split(MPI_COMM_WORLD, myrank/sqrtp, myrank, &rowComm);
   MPI_Comm_split(MPI_COMM_WORLD, myrank%sqrtp, myrank, &colComm);

   // get other rows and columns
   // TODO: merge into one loop?
   for(i=0; i<sqrtp; i++){
       MPI_Gather(a, squareSize*squareSize, MPI_DOUBLE, rowA, (n*n)/p, MPI_DOUBLE, i, rowComm);
   }
   for(i=0;i<sqrtp;i++){
       MPI_Gather(b, squareSize*squareSize, MPI_DOUBLE, colB, (n*n)/p, MPI_DOUBLE, i, colComm);
   }

   // do the actual multiplication
   for(i=0; i<squareSize; i++){
       for(j=0; j<squareSize; j++) {
           sum = 0.;
           // introduce another loop for a bit of mathematical gymnastics
           // as the blocks are gathered such that the original rows/columns of the matrices are not in order
           for(int l=0;l<sqrtp;l++){
               for(k=0; k<squareSize; k++){
                   sum += rowA[i*squareSize+k+(l*squareSize*squareSize)]*colB[k*squareSize+j+(l*squareSize*squareSize)];
               }
           }
           c[i*squareSize+j] += sum;
       }
    }

   free(rowA);
   free(colB);
   MPI_Barrier(MPI_COMM_WORLD);
   if(myrank==0)
       printf("It took %f seconds to multiply 2 %dx%d matrices.\n",
   MPI_Wtime()-start, n, n);
   if(myrank==0)
     allC = malloc(n*n*sizeof(double));

   MPI_Request sendRequestC;
   // processor with rank 0 should receive data from all other processors
   for(int i=0;i<squareSize;i++){
       MPI_Issend(&c[i*squareSize],squareSize,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&sendRequestC);
   }

   if(myrank==0){
       int currentProc=0;
       for(int i=0;i<sqrtp;i++){
           for(int j=0;j<sqrtp;j++){
               for(int k=0;k<squareSize;k++){
                   MPI_Recv(&allC[(i*squareSize*n)+(j*squareSize)+(k*n)],squareSize,MPI_DOUBLE,currentProc,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
               }
               currentProc++;
           }
       }
   }
   MPI_Wait(&sendRequestC, MPI_STATUS_IGNORE);
   //MPI_Gather(c, squareSize*squareSize, MPI_DOUBLE, allC, squareSize*squareSize, MPI_DOUBLE,0, MPI_COMM_WORLD);
   if(myrank==0) {
     for(i=0, sumdiag=0.; i<n; i++)
       sumdiag += allC[i*n+i];
     printf("The trace of the resulting matrix is %f\n", sumdiag);
    for(int i=0;i<n;i++){
        for(int j=0;j<n;j++){
            printf("%lf ", allC[i*n + j]);
        }
        printf("\n");
    }
   }
   if(myrank==0){
     free(allC);
     free(matA);
     free(matB);
   }
   MPI_Finalize();
   free(a);
   free(b);
   free(c);
}

void InitializeMatrix(int seed, double *matrix){
    srand(time(NULL));    
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
