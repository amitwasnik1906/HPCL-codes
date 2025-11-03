#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define SIZE 200

void vector_add_static(int *a, int *b, int *c, int chunk) {
    #pragma omp parallel for schedule(static, chunk)
    for (int i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }
}

void vector_add_dynamic(int *a, int *b, int *c, int chunk) {
    #pragma omp parallel for schedule(dynamic, chunk)
    for (int i = 0; i < SIZE; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int a[SIZE], b[SIZE], c[SIZE];
    for (int i = 0; i < SIZE; i++) {
        a[i] = i;
        b[i] = SIZE - i;
    }

    int chunk_sizes[] = {1, 5, 10, 25, 50};

    for (int i = 0; i < 5; i++) {
        int chunk = chunk_sizes[i];
        double start, end;

        // STATIC
        start = omp_get_wtime();
        vector_add_static(a, b, c, chunk);
        end = omp_get_wtime();
        printf("STATIC | Chunk: %d | Time: %f sec\n", chunk, end - start);

        // DYNAMIC
        start = omp_get_wtime();
        vector_add_dynamic(a, b, c, chunk);
        end = omp_get_wtime();
        printf("DYNAMIC| Chunk: %d | Time: %f sec\n", chunk, end - start);
    }

    // nowait demo
    #pragma omp parallel
    {
        #pragma omp for nowait
        for (int i = 0; i < SIZE / 2; i++) {
            c[i] = a[i] + b[i];
        }

        #pragma omp for
        for (int i = SIZE / 2; i < SIZE; i++) {
            c[i] = a[i] - b[i];
        }
    }

    return 0;
}

