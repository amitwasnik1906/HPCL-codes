#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 1000000

int cmp_asc(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

int cmp_desc(const void *a, const void *b) {
    return (*(int *)b - *(int *)a);
}

int main() {
    int *a = malloc(SIZE * sizeof(int));
    int *b = malloc(SIZE * sizeof(int));
    long long int result = 0;

    for (int i = 0; i < SIZE; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
    }

    qsort(a, SIZE, sizeof(int), cmp_asc);
    qsort(b, SIZE, sizeof(int), cmp_desc);

    double start = omp_get_wtime();

    #pragma omp parallel for reduction(+:result)
    for (int i = 0; i < SIZE; i++) {
        result += a[i] * b[i];
    }

    double end = omp_get_wtime();

    printf("Minimum Scalar Product = %lld\n", result);
    printf("Time taken: %f seconds\n", end - start);

    free(a); free(b);
    return 0;
}

