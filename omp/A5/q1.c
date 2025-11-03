// mm_mult.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int N = 1024; // adjust for testing
    double *A = malloc(N * N * sizeof(double));
    double *B = malloc(N * N * sizeof(double));
    double *C = malloc(N * N * sizeof(double));

    // init
    for (int i=0;i<N*N;i++) { A[i]=1.0; B[i]=1.0; C[i]=0.0; }

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < N; ++i) {
        for (int k = 0; k < N; ++k) {
            double a = A[i*N + k];
            for (int j = 0; j < N; ++j) {
                C[i*N + j] += a * B[k*N + j];
            }
        }
    }

    double t1 = omp_get_wtime();
    printf("Time: %f sec\n", t1 - t0);

    // checksum
    double sum = 0;
    for (int i=0;i<N*N;i++) sum += C[i];
    printf("Checksum: %f\n", sum);

    free(A); free(B); free(C);
    return 0;
}

