// mv_mult.c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int m = 20000, n = 1000;
    double *A = malloc((size_t)m * n * sizeof(double));
    double *x = malloc(n * sizeof(double));
    double *y = calloc(m, sizeof(double));

    for (int i=0;i<n;i++) x[i] = 1.0;
    for (size_t i=0;i<(size_t)m*n;i++) A[i] = 0.5;

    double t0 = omp_get_wtime();

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < m; ++i) {
        double sum = 0.0;
        size_t base = (size_t)i * n;
        for (int j = 0; j < n; ++j) sum += A[base + j] * x[j];
        y[i] = sum;
    }

    double t1 = omp_get_wtime();
    printf("Time: %f sec\n", t1 - t0);
    printf("y[0]=%f y[m-1]=%f\n", y[0], y[m-1]);

    free(A); free(x); free(y);
    return 0;
}

