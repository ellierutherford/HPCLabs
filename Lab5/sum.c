#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int p;
int n;
int main(int argc, char **argv){

   int *arr, *subarr;
   int myrank;
   int globalsum;
   n = atoi(argv[1]);
   arr = malloc(n*sizeof(int));
   for(int i=0;i<n;i++){
       arr[i] = i+1;
   }

   MPI_Init(&argc, &argv);
   MPI_Comm_size(MPI_COMM_WORLD,&p);
   MPI_Comm_rank(MPI_COMM_WORLD,&myrank);
   subarr = malloc(n/p*sizeof(int));
   MPI_Scatter(arr, n/p, MPI_INT, subarr, n/p, MPI_INT, 0, MPI_COMM_WORLD);

   int mysum = 0;
   for(int i=0; i<n/p; i++)
      mysum += subarr[i];
   printf("my rank is %d, my sum is %d\n",myrank, mysum);
   MPI_Reduce(&mysum, &globalsum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

   free(subarr);
   MPI_Finalize();
   free(arr);
   if(myrank==0)
      printf("final sum is %d\n",globalsum);

}

