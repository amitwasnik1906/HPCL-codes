/*
 * matmat_mpi.c
 * Parallel matrix-matrix multiplication using MPI
 *
 * Each process holds a block of rows of A and the full matrix B.
 * Computes local C = localA * B, then sends back to root.
 *
 * Compile:
 *   mpicc -O2 -o matmat_mpi matmat_mpi.c
 *
 * Run example:
 *   mpirun -np 4 ./matmat_mpi 1024
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/* Initialize matrix with random but deterministic values */
static double drand(int seed) {
    unsigned int x = (unsigned int) seed;
    x = (1103515245u * x + 12345u) & 0x7fffffff;
    return (double)(x % 1000) / 1000.0;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s N [validate]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);
    int validate = (argc >= 3) ? atoi(argv[2]) : 0;

    int base = N / size;
    int rem = N % size;

    int *sendcounts = malloc(size * sizeof(int));
    int *displs = malloc(size * sizeof(int));
    int *sendcounts_elems = malloc(size * sizeof(int));
    int *displs_elems = malloc(size * sizeof(int));

    int offset_rows = 0;
    for (int p = 0; p < size; p++) {
        int rows = base + (p < rem ? 1 : 0);
        sendcounts[p] = rows;
        displs[p] = offset_rows;
        sendcounts_elems[p] = rows * N;
        displs_elems[p] = offset_rows * N;
        offset_rows += rows;
    }

    int local_rows = sendcounts[rank];

    double *A = NULL;
    double *B = malloc(N * N * sizeof(double));
    double *localA = malloc(local_rows * N * sizeof(double));
    double *localC = malloc(local_rows * N * sizeof(double));
    double *C = NULL;

    if (rank == 0) {
        A = malloc(N * N * sizeof(double));
        C = malloc(N * N * sizeof(double));

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                A[i * N + j] = drand(i * N + j + 1);

        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                B[i * N + j] = drand(i + j + 12345);
    }

    // Broadcast matrix B to all processes
    MPI_Bcast(B, N * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Scatter rows of A to processes
    MPI_Scatterv(A, sendcounts_elems, displs_elems, MPI_DOUBLE,
                 localA, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Local computation: localC = localA * B
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++) {
                sum += localA[i * N + k] * B[k * N + j];
            }
            localC[i * N + j] = sum;
        }
    }

    double t1 = MPI_Wtime();
    double local_time = t1 - t0;

    // Gather results
    MPI_Gatherv(localC, local_rows * N, MPI_DOUBLE,
                C, sendcounts_elems, displs_elems, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N=%d P=%d max_compute_time=%.6f sec\n", N, size, max_time);
    }

    // Optional: Validation
    if (validate && rank == 0) {
        double *Cseq = malloc(N * N * sizeof(double));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                double s = 0.0;
                for (int k = 0; k < N; k++)
                    s += A[i * N + k] * B[k * N + j];
                Cseq[i * N + j] = s;
            }
        }
        double max_diff = 0.0;
        for (int i = 0; i < N * N; i++) {
            double diff = fabs(Cseq[i] - C[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Validation max_abs_diff = %.12e\n", max_diff);
        free(Cseq);
    }

    if (A) free(A);
    if (C) free(C);
    free(B);
    free(localA);
    free(localC);
    free(sendcounts);
    free(displs);
    free(sendcounts_elems);
    free(displs_elems);

    MPI_Finalize();
    return 0;
}
