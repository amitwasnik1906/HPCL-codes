#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define N 1000000  // Size of vectors

int main(int argc, char** argv) {
    int rank, size;
    double *A = NULL, *B = NULL;
    double local_dot = 0.0, global_dot = 0.0;
    int i, local_n;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide work among processes
    local_n = N / size;

    double *local_A = (double*)malloc(local_n * sizeof(double));
    double *local_B = (double*)malloc(local_n * sizeof(double));

    // Master process initializes the data
    if (rank == 0) {
        A = (double*)malloc(N * sizeof(double));
        B = (double*)malloc(N * sizeof(double));

        srand(time(NULL));
        for (i = 0; i < N; i++) {
            A[i] = rand() % 100;
            B[i] = rand() % 100;
        }
    }

    // Scatter data to all processes
    MPI_Scatter(A, local_n, MPI_DOUBLE, local_A, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, local_n, MPI_DOUBLE, local_B, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Start timer
    double start = MPI_Wtime();

    // Compute local dot product
    for (i = 0; i < local_n; i++)
        local_dot += local_A[i] * local_B[i];

    // Reduce all partial results to get the global dot product
    MPI_Reduce(&local_dot, &global_dot, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Stop timer
    double end = MPI_Wtime();

    // Master prints result and execution time
    if (rank == 0) {
        printf("Dot Product = %f\n", global_dot);
        printf("Execution Time: %f seconds with %d processes\n", end - start, size);
    }

    // Free memory
    free(local_A);
    free(local_B);
    if (rank == 0) {
        free(A);
        free(B);
    }

    MPI_Finalize();
    return 0;
}
