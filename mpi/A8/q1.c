/*
 * conv2d_mpi.c
 * Parallel 2D Convolution using MPI (row-wise decomposition)
 *
 * Compile:
 *   mpicc -O2 -o conv2d_mpi conv2d_mpi.c
 *
 * Run Example:
 *   mpirun -np 4 ./conv2d_mpi 512 3
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define IDX(i, j, N) ((i) * (N) + (j))

void random_matrix(double *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = (double)(rand() % 10);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <image_size N> <kernel_size M>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int N = atoi(argv[1]);  // Image size N x N
    int M = atoi(argv[2]);  // Kernel size M x M
    int pad = M / 2;

    double *image = NULL;
    double *kernel = malloc(M * M * sizeof(double));

    // Initialize kernel
    if (rank == 0) {
        image = malloc(N * N * sizeof(double));
        random_matrix(image, N, N);
        random_matrix(kernel, M, M);
    }

    // Broadcast kernel to all processes
    MPI_Bcast(kernel, M * M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Determine local rows (with overlap)
    int base = N / size;
    int rem = N % size;
    int start_row = rank * base + (rank < rem ? rank : rem);
    int local_rows = base + (rank < rem ? 1 : 0);

    // Add padding rows (halo)
    int local_rows_with_halo = local_rows + 2 * pad;
    double *local_image = malloc(local_rows_with_halo * N * sizeof(double));
    double *local_output = malloc(local_rows * N * sizeof(double));

    // Send counts and displacements for scatterv
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));

        int offset = 0;
        for (int p = 0; p < size; p++) {
            int rows = base + (p < rem ? 1 : 0);
            sendcounts[p] = rows * N;
            displs[p] = offset;
            offset += rows * N;
        }
    }

    // Scatter image (each process gets its part)
    MPI_Scatterv(image, sendcounts, displs, MPI_DOUBLE,
                 &local_image[pad * N], local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // Exchange halo rows
    MPI_Status status;
    if (rank > 0)
        MPI_Sendrecv(&local_image[pad * N], pad * N, MPI_DOUBLE, rank - 1, 0,
                     local_image, pad * N, MPI_DOUBLE, rank - 1, 0,
                     MPI_COMM_WORLD, &status);
    if (rank < size - 1)
        MPI_Sendrecv(&local_image[(local_rows) * N], pad * N, MPI_DOUBLE, rank + 1, 0,
                     &local_image[(local_rows + pad) * N], pad * N, MPI_DOUBLE, rank + 1, 0,
                     MPI_COMM_WORLD, &status);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    // Perform convolution on local rows
    for (int i = pad; i < local_rows + pad; i++) {
        for (int j = pad; j < N - pad; j++) {
            double sum = 0.0;
            for (int u = 0; u < M; u++) {
                for (int v = 0; v < M; v++) {
                    int x = i + u - pad;
                    int y = j + v - pad;
                    sum += local_image[IDX(x, y, N)] * kernel[IDX(u, v, M)];
                }
            }
            local_output[IDX(i - pad, j, N)] = sum;
        }
    }

    double local_time = MPI_Wtime() - t0;

    // Gather results
    double *output = NULL;
    if (rank == 0)
        output = malloc(N * N * sizeof(double));

    MPI_Gatherv(local_output, local_rows * N, MPI_DOUBLE,
                output, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Image Size: %dx%d, Kernel: %dx%d, Processes: %d\n", N, N, M, M, size);
        printf("Max compute time: %.6f sec\n", max_time);
    }

    if (rank == 0) {
        free(image);
        free(output);
        free(sendcounts);
        free(displs);
    }
    free(kernel);
    free(local_image);
    free(local_output);

    MPI_Finalize();
    return 0;
}
