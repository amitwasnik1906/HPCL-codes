#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define SIZE 10000000  // Default data size

int main() {
    int *a = malloc(SIZE * sizeof(int));
    int *b = malloc(SIZE * sizeof(int));
    int scalar = 5;

    // Initialize vector a
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
    }

    double start = omp_get_wtime();

    // Vector-scalar addition using OpenMP
    #pragma omp parallel for
    for (int i = 0; i < SIZE; i++) {
        b[i] = a[i] + scalar;
    }

    double end = omp_get_wtime();
    printf("Time taken: %f seconds\n", end - start);

    // Optional: print first 10 results
    for (int i = 0; i < 10; i++) {
        printf("b[%d] = %d\n", i, b[i]);
    }

    free(a);
    free(b);
    return 0;
}

