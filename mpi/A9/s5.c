/* spec_matmul_mpi.c
   Compile: mpicc -O2 -o spec_matmul_mpi spec_matmul_mpi.c
   Run: mpirun -np 3 ./spec_matmul_mpi N
   Ranks:
     0 - coordinator
     1 - Strassen (method A, "approx"/fast)
     2 - Classical (method B, exact)
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TAG_DATA 10
#define TAG_DONE 20
#define TAG_CANCEL 30
#define TAG_RESULT 40

/* Helper: allocate NxN double matrix contiguous */
double *alloc_mat(int N){ return (double*)calloc((size_t)N*N, sizeof(double)); }

/* naive classical multiply with periodic cancel check */
void classical_mul_check(int N, double *A, double *B, double *C, MPI_Comm comm) {
    int cancel=0;
    for (int i=0;i<N;i++){
        for (int k=0;k<N;k++){
            double aik = A[i*N + k];
            for (int j=0;j<N;j++){
                C[i*N + j] += aik * B[k*N + j];
            }
        }
        /* check cancel every row */
        MPI_Iprobe(0, TAG_CANCEL, comm, &cancel, MPI_STATUS_IGNORE);
        if (cancel) {
            int flag; MPI_Recv(&flag, 1, MPI_INT, 0, TAG_CANCEL, comm, MPI_STATUS_IGNORE);
            return;
        }
    }
}

/* classical multiply (no checks) used for small blocks in Strassen */
void classical_mul_block(int n, double *A, double *B, double *C){
    for (int i=0;i<n;i++)
      for (int k=0;k<n;k++){
        double aik = A[i*n + k];
        for (int j=0;j<n;j++) C[i*n + j] += aik * B[k*n + j];
      }
}

/* add/sub helpers */
void add_block(int n, double *A, double *B, double *C){
    for (int i=0;i<n*n;i++) C[i] = A[i] + B[i];
}
void sub_block(int n, double *A, double *B, double *C){
    for (int i=0;i<n*n;i++) C[i] = A[i] - B[i];
}

/* Strassen with cooperative cancel check: checks cancel at recursion entry */
void strassen_rec(int n, double *A, double *B, double *C, MPI_Comm comm) {
    int cancel=0;
    /* Periodically check for cancel on recursion entry */
    MPI_Iprobe(0, TAG_CANCEL, comm, &cancel, MPI_STATUS_IGNORE);
    if (cancel) { int flag; MPI_Recv(&flag,1,MPI_INT,0,TAG_CANCEL,comm,MPI_STATUS_IGNORE); return; }

    if (n <= 64) { /* base case threshold */
        classical_mul_block(n,A,B,C);
        return;
    }
    int m = n/2;
    size_t bytes = (size_t)m*m*sizeof(double);
    /* allocate temporaries */
    double *A11 = A;
    double *A12 = A + m;
    double *A21 = A + m*n;
    double *A22 = A + m*n + m;
    double *B11 = B;
    double *B12 = B + m;
    double *B21 = B + m*n;
    double *B22 = B + m*n + m;
    double *C11 = C;
    double *C12 = C + m;
    double *C21 = C + m*n;
    double *C22 = C + m*n + m;

    double *S1 = (double*)malloc(bytes), *S2 = (double*)malloc(bytes), *S3 = (double*)malloc(bytes);
    double *P1 = (double*)malloc(bytes), *P2 = (double*)malloc(bytes), *P3 = (double*)malloc(bytes);
    double *P4 = (double*)malloc(bytes), *P5 = (double*)malloc(bytes), *P6 = (double*)malloc(bytes), *P7 = (double*)malloc(bytes);
    /* compute S and P matrices (standard Strassen) */
    // P1 = A11 * (B12 - B22)
    sub_block(m, B12, B22, S1); memset(P1,0,bytes); strassen_rec(m, A11, S1, P1, comm);
    // P2 = (A11 + A12) * B22
    add_block(m, A11, A12, S2); memset(P2,0,bytes); strassen_rec(m, S2, B22, P2, comm);
    // P3 = (A21 + A22) * B11
    add_block(m, A21, A22, S3); memset(P3,0,bytes); strassen_rec(m, S3, B11, P3, comm);
    // P4 = A22 * (B21 - B11)
    sub_block(m, B21, B11, S1); memset(P4,0,bytes); strassen_rec(m, A22, S1, P4, comm);
    // P5 = (A11 + A22) * (B11 + B22)
    add_block(m, A11, A22, S2); add_block(m, B11, B22, S3); memset(P5,0,bytes); strassen_rec(m, S2, S3, P5, comm);
    // P6 = (A12 - A22) * (B21 + B22)
    sub_block(m, A12, A22, S2); add_block(m, B21, B22, S3); memset(P6,0,bytes); strassen_rec(m, S2, S3, P6, comm);
    // P7 = (A11 - A21) * (B11 + B12)
    sub_block(m, A11, A21, S2); add_block(m, B11, B12, S3); memset(P7,0,bytes); strassen_rec(m, S2, S3, P7, comm);

    /* assemble C blocks */
    // C11 = P5 + P4 - P2 + P6
    for (int i=0;i<m*m;i++) C11[i] = P5[i] + P4[i] - P2[i] + P6[i];
    // C12 = P1 + P2
    for (int i=0;i<m*m;i++) C12[i] = P1[i] + P2[i];
    // C21 = P3 + P4
    for (int i=0;i<m*m;i++) C21[i] = P3[i] + P4[i];
    // C22 = P5 + P1 - P3 - P7
    for (int i=0;i<m*m;i++) C22[i] = P5[i] + P1[i] - P3[i] - P7[i];

    free(S1); free(S2); free(S3);
    free(P1); free(P2); free(P3); free(P4); free(P5); free(P6); free(P7);
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (size < 3) {
        if (rank==0) fprintf(stderr,"Run with -np 3\n");
        MPI_Finalize(); return 1;
    }
    int N = 256;
    if (argc > 1) N = atoi(argv[1]);
    if (rank == 0) {
        /* coordinator */
        srand((unsigned)time(NULL));
        double *A = alloc_mat(N), *B = alloc_mat(N);
        for (int i=0;i<N*N;i++){ A[i] = (rand()%100)/10.0; B[i] = (rand()%100)/10.0; }
        /* send N and data to workers */
        for (int r=1;r<=2;r++){
            MPI_Send(&N,1,MPI_INT,r,TAG_DATA,MPI_COMM_WORLD);
            MPI_Send(A,N*N,MPI_DOUBLE,r,TAG_DATA,MPI_COMM_WORLD);
            MPI_Send(B,N*N,MPI_DOUBLE,r,TAG_DATA,MPI_COMM_WORLD);
        }
        free(A); free(B);

        /* wait for first DONE from either worker */
        int winner; MPI_Status st;
        MPI_Recv(&winner,1,MPI_INT,MPI_ANY_SOURCE,TAG_DONE,MPI_COMM_WORLD,&st);
        int src = st.MPI_SOURCE;
        printf("[coord] received DONE from rank %d\n", src);
        /* tell the other to cancel */
        int other = (src==1)?2:1;
        int canc = 1; MPI_Send(&canc,1,MPI_INT,other,TAG_CANCEL,MPI_COMM_WORLD);

        /* receive result size then matrix result from winner */
        int recvN; MPI_Recv(&recvN,1,MPI_INT,src,TAG_RESULT,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        double *C = alloc_mat(recvN);
        MPI_Recv(C, recvN*recvN, MPI_DOUBLE, src, TAG_RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("[coord] received result from winner %d (N=%d). Sample C[0]=%f\n", src, recvN, C[0]);
        free(C);
    } else {
        /* worker: receive N and data */
        int n; MPI_Recv(&n,1,MPI_INT,0,TAG_DATA,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        double *A = alloc_mat(n), *B = alloc_mat(n);
        MPI_Recv(A,n*n,MPI_DOUBLE,0,TAG_DATA,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(B,n*n,MPI_DOUBLE,0,TAG_DATA,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        double *C = alloc_mat(n);
        int cancel=0;
        MPI_Request req_cancel;
        MPI_Status st;

        if (rank == 1) {
            /* Strassen worker */
            double t0 = MPI_Wtime();
            strassen_rec(n,A,B,C,MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            /* before announcing done, check if cancel came in */
            MPI_Iprobe(0,TAG_CANCEL,MPI_COMM_WORLD,&cancel,&st);
            if (!cancel) {
                int done_msg = 1; MPI_Send(&done_msg,1,MPI_INT,0,TAG_DONE,MPI_COMM_WORLD);
                /* send result */
                MPI_Send(&n,1,MPI_INT,0,TAG_RESULT,MPI_COMM_WORLD);
                MPI_Send(C,n*n,MPI_DOUBLE,0,TAG_RESULT,MPI_COMM_WORLD);
                printf("[rank1] finished Strassen in %g s, sent result\n", t1-t0);
            } else {
                /* consume cancel */
                int f; MPI_Recv(&f,1,MPI_INT,0,TAG_CANCEL,MPI_COMM_WORLD,&st);
                printf("[rank1] cancelled before send\n");
            }
        } else if (rank == 2) {
            /* Classical worker */
            double t0 = MPI_Wtime();
            classical_mul_check(n,A,B,C,MPI_COMM_WORLD);
            double t1 = MPI_Wtime();
            /* check whether cancel arrived */
            MPI_Iprobe(0,TAG_CANCEL,MPI_COMM_WORLD,&cancel,&st);
            if (!cancel) {
                int done_msg = 1; MPI_Send(&done_msg,1,MPI_INT,0,TAG_DONE,MPI_COMM_WORLD);
                MPI_Send(&n,1,MPI_INT,0,TAG_RESULT,MPI_COMM_WORLD);
                MPI_Send(C,n*n,MPI_DOUBLE,0,TAG_RESULT,MPI_COMM_WORLD);
                printf("[rank2] finished Classical in %g s, sent result\n", t1-t0);
            } else {
                int f; MPI_Recv(&f,1,MPI_INT,0,TAG_CANCEL,MPI_COMM_WORLD,&st);
                printf("[rank2] cancelled before send\n");
            }
        }
        free(A); free(B); free(C);
    }
    MPI_Finalize();
    return 0;
}
