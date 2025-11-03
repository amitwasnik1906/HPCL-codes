// mm_scalar.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int rows = 4000, cols = 4000;
    double scalar = 2.5;
    double *A = malloc(rows * cols * sizeof(double));
    double *B = malloc(rows * cols * sizeof(double));

    for (int i=0;i<rows*cols;i++) A[i] = i % 100;

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int idx = 0; idx < rows * cols; ++idx) {
        B[idx] = scalar * A[idx];
    }

    double t1 = omp_get_wtime();
    printf("Time: %f sec\n", t1 - t0);
    printf("Sample B[0]=%f B[last]=%f\n", B[0], B[rows*cols-1]);

    free(A); free(B);
    return 0;
}

