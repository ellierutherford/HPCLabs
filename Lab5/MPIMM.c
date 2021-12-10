

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
   double matB[] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36};
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
   
   MPI_Request recvRequest;
   MPI_Request sendRequest;
   if(myrank==0){
   int currentProc=0;
   for(int i=0;i<sqrtp;i++){
       for(int j=0;j<sqrtp;j++){
           printf("current proc is %d\n",currentProc); 
           for(int k=0;k<squareSize;k++){
               MPI_Issend(&matA[(i*squareSize*n)+(j*squareSize)+(k*n)],squareSize,MPI_DOUBLE,currentProc,1,MPI_COMM_WORLD,&sendRequest);
           }
           currentProc++;
       }
   }
   }

   for(int i=0;i<p;i++){
       for(int j=0;j<squareSize;j++){
           MPI_Irecv(a,squareSize,MPI_DOUBLE,0,1,MPI_COMM_WORLD,&recvRequest);
       }
   }
   //MPI_Wait(&recvRequest, MPI_STATUS_IGNORE);
   //MPI_Type_create_resized(col, 0, 1*sizeof(double), &coltype);
   //MPI_Type_commit(&coltype);
   /*for(i=0; i<squareSize*squareSize; i++) {
     a[i] = 1.;
     b[i] = 2.;
   }*/
   //MPI_Scatter(matA, squareSize*squareSize, MPI_DOUBLE, a, squareSize*squareSize, MPI_DOUBLE,myrank,MPI_COMM_WORLD);
   //MPI_Scatter(matrix, part_size, coltype, mypart, part_size, coltype, 0, MPI_COMM_WORLD);
   MPI_Scatter(matB, squareSize*squareSize, MPI_DOUBLE, b, squareSize*squareSize, MPI_DOUBLE, myrank, MPI_COMM_WORLD);
   /*printf("A: rank is %d:\n",myrank);
   for(int i=0;i<squareSize*squareSize;i++){
       printf("%lf ", a[i]);
   }*/
   printf("\n");
   printf("A: rank is %d:\n",myrank);
   for(int i=0;i<squareSize*squareSize;i++){
       printf("%lf ", a[i]);
   }
   printf("\n");
   MPI_Barrier(MPI_COMM_WORLD);
   if(myrank==0)
     start = MPI_Wtime();

   int colour = myrank/sqrtp;
   //printf("colour for processor %d is %d\n", myrank,colour);

   MPI_Comm_split(MPI_COMM_WORLD, colour, myrank, &rowComm);
   MPI_Comm_split(MPI_COMM_WORLD,myrank%sqrtp,myrank,&colComm);

   // TODO: merge into one loop?
   for(i=0; i<sqrtp; i++){
       MPI_Gather(a, squareSize*squareSize, MPI_DOUBLE, rowA, (n*n)/p, MPI_DOUBLE, i, rowComm);
   }
   for(i=0;i<sqrtp;i++){
       MPI_Gather(b, squareSize*squareSize, MPI_DOUBLE, colB, (n*n)/p, MPI_DOUBLE, i, colComm);
   }

   for(int l=0;l<squareSize*n;l++){
       printf("rank is %d, element %d of rowA is %lf \n",myrank,l,rowA[l]);
       //if(l=n*n-1){
         //printf("\n");
       //}
   }
   for(int l=0;l<squareSize*n;l++){
       printf("rank is %d, element %d of colB is %lf \n",myrank,l,colB[l]);
   }

   /*for(int l=0;l<squareSize*n;l++){
      printf("%lf ",colB[l]);
   }*/


   for(i=0; i<squareSize; i++){
       for(j=0; j<squareSize; j++) {
           sum = 0.;
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

