#include <stdio.h>
#include <omp.h>

int main() {
    int n;

    printf("Enter the number of threads: ");
    scanf("%d", &n);

    printf("\n--- Sequential Output ---\n");
    for (int i = 0; i < n; i++) {
        printf("Hello, World from thread %d (sequential)\n", i);
    }

    printf("\n--- Parallel Output ---\n");

    omp_set_num_threads(n);

    #pragma omp parallel
    {
        int id = omp_get_thread_num();
        printf("Hello, World from thread %d (parallel)\n", id);
    }

    return 0;
}

