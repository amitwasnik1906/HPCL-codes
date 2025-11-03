#include <stdio.h>
#include <omp.h>

int main() {
    long long int num_steps = 100000000;
    double step = 1.0 / (double)num_steps;
    double pi = 0.0;

    double start = omp_get_wtime();

    #pragma omp parallel
    {
        double x, sum = 0.0;
        #pragma omp for
        for (int i = 0; i < num_steps; i++) {
            x = (i + 0.5) * step;
            sum += 4.0 / (1.0 + x * x);
        }

        #pragma omp atomic
        pi += sum;
    }

    pi *= step;

    double end = omp_get_wtime();

    printf("Calculated Pi = %.15f\n", pi);
    printf("Time taken = %f seconds\n", end - start);

    return 0;
}

