#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            printf("Please run with 2 processes: mpirun -np 2 ./a.out\n");
        MPI_Finalize();
        return 0;
    }

    int n = 8; // Example size
    int A[8] = {1, 2, 3, 4, 5, 6, 7, 8};

    int local_sum = 0;

    if (rank == 0) {
        for (int i = 0; i < n/2; i++)
            local_sum += A[i];
    } else {
        for (int i = n/2; i < n; i++)
            local_sum += A[i];
    }

    int total_sum;
    MPI_Reduce(&local_sum, &total_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Final Sum = %d\n", total_sum);

    MPI_Finalize();
    return 0;
}
