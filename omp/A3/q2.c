#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void matrix_add(int **a, int **b, int **c, int size) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            c[i][j] = a[i][j] + b[i][j];
}

int **alloc_matrix(int size) {
    int **mat = malloc(size * sizeof(int *));
    for (int i = 0; i < size; i++)
        mat[i] = malloc(size * sizeof(int));
    return mat;
}

void free_matrix(int **mat, int size) {
    for (int i = 0; i < size; i++)
        free(mat[i]);
    free(mat);
}

int main() {
    int sizes[] = {250, 500, 750, 1000, 2000};

    for (int s = 0; s < 5; s++) {
        int size = sizes[s];
        printf("\nMatrix Size: %d x %d\n", size, size);

        int **a = alloc_matrix(size);
        int **b = alloc_matrix(size);
        int **c = alloc_matrix(size);

        // Initialize matrices
        for (int i = 0; i < size; i++)
            for (int j = 0; j < size; j++) {
                a[i][j] = rand() % 100;
                b[i][j] = rand() % 100;
            }

        for (int threads = 1; threads <= 8; threads *= 2) {
            omp_set_num_threads(threads);
            double start = omp_get_wtime();
            matrix_add(a, b, c, size);
            double end = omp_get_wtime();

            printf("Threads: %d, Time: %f sec\n", threads, end - start);
        }

        free_matrix(a, size); free_matrix(b, size); free_matrix(c, size);
    }

    return 0;
}

