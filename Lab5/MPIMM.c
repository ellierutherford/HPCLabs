#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <mpi.h>
#include <math.h>

int n, p;
int main(int argc, char **argv) {
   int myn, myrank;
   double *a, *b, *rowA, *colB, start, sum, sumdiag;
   int i, j, k;
   n = atoi(argv[1]);
   MPI_Comm rowComm, colComm;
   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD,&p);
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   double sqrtp = sqrt(p);
   double numOfSquares = p;
   double squareSize = n/sqrtp;
   printf("squareSize is %lf, n is %d \n", squareSize,n);
   a = malloc(squareSize*squareSize*sizeof(double));
   b = malloc(squareSize*squareSize*sizeof(double));
   //c = malloc(squareSize*squareSize*sizeof(double));
   //rowA = malloc(n*sqrt(squareSize)*sizeof(double));
   //colB = malloc(n*sqrt(squareSize)*sizeof(double));
   rowA = malloc(squareSize*n*sizeof(double));
   colB = malloc(squareSize*n*sizeof(double));
   for(i=0; i<squareSize*squareSize; i++) {
     a[i] = 1.;
     b[i] = 2.;
   }
   MPI_Barrier(MPI_COMM_WORLD);
   /*if(myrank==0)
     start = MPI_Wtime();*/

   int color = myrank/n;
   MPI_Comm_split(MPI_COMM_WORLD,color,myrank,&rowComm);
   //MPI_Comm_split(MPI_COMM_WORLD,color,myrank,&colComm);
   
   for(i=0; i<p; i++){
       MPI_Gather(a, squareSize*squareSize, MPI_DOUBLE, rowA, (n*n)/p, MPI_DOUBLE, i, rowComm);
   }
   MPI_Comm_split(MPI_COMM_WORLD,color,myrank,&colComm);
   for(i=0;i<p;i++){
       MPI_Gather(b, squareSize*squareSize, MPI_DOUBLE, colB, (n*n)/p, MPI_DOUBLE, i, colComm);
   }

   for(int l=0;l<squareSize*n;l++){
       printf("%lf ",rowA[l]);
       //if(l=n*n-1){
         //printf("\n");
       //}
   }
   free(rowA);
   free(colB);
   MPI_Barrier(MPI_COMM_WORLD);
   /*if(myrank==0)
       printf("It took %f seconds to multiply 2 %dx%d matrices.\n",
   MPI_Wtime()-start, n, n);
   /*if(myrank==0)
     allC = malloc(n*n*sizeof(double));
   MPI_Gather(c, squareSize*squareSize, MPI_DOUBLE, allC, squareSize*squareSize, MPI_DOUBLE,0, MPI_COMM_WORLD);
   if(myrank==0) {
     for(i=0, sumdiag=0.; i<n; i++)
       sumdiag += allC[i*n+i];
     printf("The trace of the resulting matrix is %f\n", sumdiag);
   }
   if(myrank==0)
     free(allC);*/
   MPI_Finalize();
   free(a);
   free(b);
   //free(c);
}

