/*
 * matvec_mpi.c
 * Parallel matrix-vector multiplication using MPI.
 *
 * Each process gets a contiguous block of rows of A,
 * the full vector x is broadcast to everyone.
 *
 * Build: mpicc -O2 -o matvec_mpi matvec_mpi.c
 * Run example: mpirun -np 4 ./matvec_mpi 4096
 *
 * Arguments: ./matvec_mpi N [validate]
 *   N = matrix dimension (N x N)
 *   validate = optional 1 to run a sequential check (default 0)
 *
 * Uses MPI_Scatterv / MPI_Gatherv so that N doesn't have to be divisible by P.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Simple random init (deterministic) */
static double drand(int seed) {
    unsigned int x = (unsigned int) seed;
    x = (1103515245u * x + 12345u) & 0x7fffffff;
    return (double)(x % 1000) / 1000.0;
}

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 2) {
        if (rank == 0) fprintf(stderr, "Usage: %s N [validate]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }

    const int N = atoi(argv[1]);
    const int validate = (argc >= 3) ? atoi(argv[2]) : 0;

    /* compute row counts per process */
    int base = N / size;
    int rem = N % size;
    int *sendcounts = (int*) malloc(size * sizeof(int)); /* number of rows */
    int *displs = (int*) malloc(size * sizeof(int));     /* displacement in rows */
    int *sendcounts_elems = (int*) malloc(size * sizeof(int)); /* number of elements (rows*N) */
    int *displs_elems = (int*) malloc(size * sizeof(int));     /* displacement in elements */

    int offset_rows = 0;
    for (int p = 0; p < size; ++p) {
        int rows = base + (p < rem ? 1 : 0);
        sendcounts[p] = rows;
        displs[p] = offset_rows;
        sendcounts_elems[p] = rows * N;
        displs_elems[p] = offset_rows * N;
        offset_rows += rows;
    }

    /* Local rows for this process */
    int local_rows = sendcounts[rank];

    /* Buffers:
     * - local_A: local_rows x N
     * - x: N
     * - local_y: local_rows
     */
    double *local_A = NULL;
    if (local_rows > 0) {
        local_A = (double*) malloc((size_t) local_rows * N * sizeof(double));
        if (!local_A) { fprintf(stderr, "alloc local_A failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    double *x = (double*) malloc((size_t) N * sizeof(double));
    double *local_y = (double*) malloc((size_t) local_rows * sizeof(double));
    if (!x || !local_y) { fprintf(stderr, "alloc x/local_y failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }

    /* Root constructs the matrix and vector (deterministic values for repeatability) */
    double *A = NULL;
    if (rank == 0) {
        A = (double*) malloc((size_t) N * N * sizeof(double));
        if (!A) { fprintf(stderr, "alloc A failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                A[i*N + j] = drand(i * N + j + 1);
            }
            x[i] = drand(i + 12345);
        }
    }

    /* Broadcast vector x to all processes. Root has x, others receive */
    /* First root must fill x; other processes' x content is undefined until broadcast */
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Scatter matrix A row-blocks to local_A (using scatterv with element counts sendcounts_elems) */
    MPI_Scatterv(A, sendcounts_elems, displs_elems, MPI_DOUBLE,
                 local_A, local_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    /* Synchronize and time local multiplication (exclude init time if needed) */
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();

    /* Local mat-vec: for each local row i, compute dot product of row with x */
    for (int i = 0; i < local_rows; ++i) {
        double sum = 0.0;
        double *row = &local_A[i * N];
        /* unrolled loop could be added for speed */
        for (int j = 0; j < N; ++j) sum += row[j] * x[j];
        local_y[i] = sum;
    }

    double t1 = MPI_Wtime();
    double local_compute_time = t1 - t0;

    /* Gather results local_y into y at root */
    double *y = NULL;
    if (rank == 0) {
        y = (double*) malloc((size_t) N * sizeof(double));
        if (!y) { fprintf(stderr, "alloc y failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }

    MPI_Gatherv(local_y, local_rows, MPI_DOUBLE,
                y, sendcounts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    /* Optional: measure a wall-clock time for whole operation including communication:
       do a barrier and measure from before scatterv to after gatherv if desired.
       Here we measured only local compute time — but we can compute global max compute time. */

    double max_compute_time;
    MPI_Reduce(&local_compute_time, &max_compute_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("N=%d P=%d max_compute_time=%.6f sec\n", N, size, max_compute_time);
    }

    /* Optional validation: compute sequential result on root and compare */
    if (validate && rank == 0) {
        double *y_seq = (double*) malloc((size_t) N * sizeof(double));
        for (int i = 0; i < N; ++i) {
            double s = 0.0;
            for (int j = 0; j < N; ++j) s += A[i*N + j] * x[j];
            y_seq[i] = s;
        }
        /* compare vector y and y_seq (root has both) */
        double max_diff = 0.0;
        for (int i = 0; i < N; ++i) {
            double diff = fabs(y_seq[i] - y[i]);
            if (diff > max_diff) max_diff = diff;
        }
        printf("Validation max_abs_diff = %.12e\n", max_diff);
        free(y_seq);
    }

    /* cleanup */
    free(sendcounts);
    free(displs);
    free(sendcounts_elems);
    free(displs_elems);
    if (A) free(A);
    if (local_A) free(local_A);
    free(x);
    free(local_y);
    if (y) free(y);

    MPI_Finalize();
    return EXIT_SUCCESS;
}
