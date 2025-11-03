#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);                  
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);    
    MPI_Comm_size(MPI_COMM_WORLD, &size);   

    if (size != 10) {
        if (rank == 0) {
            printf("Please run with 10 processes: mpirun -np 10 ./a.out\n");
        }
        MPI_Finalize();
        return 0;
    }

    printf("Hello World from process %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
