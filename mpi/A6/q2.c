#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);  

    if (size != 5) {
        if (rank == 0) {
            printf("Please run with 5 processes: mpirun -np 5 ./a.out\n");
        }
        MPI_Finalize();
        return 0;
    }

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    printf("Process %d belongs to communicator of size %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
