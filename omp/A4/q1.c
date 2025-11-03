/* q1.c
   Compile: gcc -fopenmp q1.c -o fib
   Run: ./fib <n> <num_threads>
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

long long *memo;
omp_lock_t *locks;

long long fib_task(int n) {
    if (n <= 1) return n;

    
    long long v = memo[n];
    if (v != -1) return v;

    
    omp_set_lock(&locks[n]);
    
    if (memo[n] != -1) {
        v = memo[n];
        omp_unset_lock(&locks[n]);
        return v;
    }

    long long x = 0, y = 0;
    
    if (n > 20) { 
        #pragma omp task shared(x) firstprivate(n)
        x = fib_task(n-1);
        #pragma omp task shared(y) firstprivate(n)
        y = fib_task(n-2);
        #pragma omp taskwait
    } else {
        x = fib_task(n-1);
        y = fib_task(n-2);
    }

    memo[n] = x + y;
    v = memo[n];
    omp_unset_lock(&locks[n]);
    return v;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <n> <num_threads>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    memo = malloc((n+1) * sizeof(long long));
    locks = malloc((n+1) * sizeof(omp_lock_t));
    for (int i=0;i<=n;i++) {
        memo[i] = -1;
        omp_init_lock(&locks[i]);
    }
    memo[0] = 0;
    if (n>=1) memo[1] = 1;

    omp_set_num_threads(num_threads);
    double t0 = omp_get_wtime();
    #pragma omp parallel
    {
        #pragma omp single
        {
            long long res = fib_task(n);
            double t1 = omp_get_wtime();
            printf("Fibonacci(%d) = %lld\n", n, res);
            printf("Threads used: %d\n", num_threads);
            printf("Time (s): %f\n", t1 - t0);
        }
    }

    for (int i=0;i<=n;i++) omp_destroy_lock(&locks[i]);
    free(memo);
    free(locks);
    return 0;
}

