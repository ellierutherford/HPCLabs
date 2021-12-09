#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <math.h>

int n, p;
int main(int argc, char **argv) {
   int myn, myrank;
   double *a, *b, *c, *rowA, *colB, start, sum, *allC, sumdiag;
   int i, j, k;
   double matA[] = {1,4,8,6,3,5,8,10,12,2,4,3,7,1,8,9,10,11,3,4,5,2,8,1,7,6,9,8,6,7,2,2,1,3,8,9};
   double matB[] = {8,6,3,4,5,2,1,3,8,3,1,5,5,1,3,5,4,5,7,6,9,2,4,6,6,4,2,3,4,5,8,9,7,1,2,3};
   n = atoi(argv[1]);
   MPI_Comm rowComm, colComm;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD,&p);
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   
   int sqrtp = (int)sqrt(p);
   int numOfSquares = p;
   int squareSize = n/sqrtp;

   a = malloc(squareSize*squareSize*sizeof(double));
   b = malloc(squareSize*squareSize*sizeof(double));
   c = malloc(squareSize*squareSize*sizeof(double));
   rowA = malloc(squareSize*n*sizeof(double));
   colB = malloc(squareSize*n*sizeof(double));
   /*for(i=0; i<squareSize*squareSize; i++) {
     a[i] = 1.;
     b[i] = 2.;
   }*/
   MPI_Scatter(matA, squareSize*squareSize, MPI_DOUBLE, a, squareSize*squareSize, MPI_DOUBLE,myrank,MPI_COMM_WORLD);
   MPI_Scatter(matB, squareSize*squareSize, MPI_DOUBLE, b, squareSize*squareSize, MPI_DOUBLE,myrank,MPI_COMM_WORLD);
   printf("A: rank is %d:\n",myrank);
   for(int i=0;i<squareSize*squareSize;i++){
       printf("%lf ", a[i]);
   }
   printf("\n");
   printf("B: rank is %d:\n",myrank);
   for(int i=0;i<squareSize*squareSize;i++){
       printf("%lf ", b[i]);
   }
   printf("\n");
   MPI_Barrier(MPI_COMM_WORLD);
   if(myrank==0)
     start = MPI_Wtime();

   int colour = myrank/sqrtp;
   printf("colour for processor %d is %d\n", myrank,colour);

   MPI_Comm_split(MPI_COMM_WORLD, colour, myrank, &rowComm);
   MPI_Comm_split(MPI_COMM_WORLD,myrank%sqrtp,myrank,&colComm);

   // TODO: merge into one loop?
   for(i=0; i<sqrtp; i++){
       MPI_Gather(a, squareSize*squareSize, MPI_DOUBLE, rowA, (n*n)/p, MPI_DOUBLE, i, rowComm);
   }
   for(i=0;i<sqrtp;i++){
       MPI_Gather(b, squareSize*squareSize, MPI_DOUBLE, colB, (n*n)/p, MPI_DOUBLE, i, colComm);
   }

   /*for(int l=0;l<squareSize*n;l++){
       printf("%lf ",rowA[l]);
       //if(l=n*n-1){
         //printf("\n");
       //}
   }

   for(int l=0;l<squareSize*n;l++){
      printf("%lf ",colB[l]);
   }*/

   for(i=0; i<squareSize; i++)
       for(j=0; j<squareSize; j++) {
           sum = 0.;
           for(k=0; k<squareSize; k++)
               sum += a[i*squareSize+k]*b[k*squareSize+j];
           c[i*squareSize+j] = sum;
        }
   
   free(rowA);
   free(colB);
   MPI_Barrier(MPI_COMM_WORLD);
   /*if(myrank==0)
       printf("It took %f seconds to multiply 2 %dx%d matrices.\n",
   MPI_Wtime()-start, n, n);*/
   if(myrank==0)
     allC = malloc(n*n*sizeof(double));
   MPI_Gather(c, squareSize*squareSize, MPI_DOUBLE, allC, squareSize*squareSize, MPI_DOUBLE,0, MPI_COMM_WORLD);
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
   if(myrank==0)
     free(allC);
   MPI_Finalize();
   free(a);
   free(b);
   free(c);
}

