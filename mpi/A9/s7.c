/* spec_sort_mpi.c
   Compile: mpicc -O2 -o spec_sort_mpi spec_sort_mpi.c
   Run: mpirun -np 3 ./spec_sort_mpi N
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* tags */
#define TAG_ARR 101
#define TAG_DONE 102
#define TAG_CANCEL 103
#define TAG_RESULT 104

/* Merge sort */
void merge(long *a, int l, int m, int r){
    int n1 = m - l, n2 = r - m;
    long *L = malloc(n1*sizeof(long)); long *R = malloc(n2*sizeof(long));
    for (int i=0;i<n1;i++) L[i]=a[l+i];
    for (int j=0;j<n2;j++) R[j]=a[m+j];
    int i=0,j=0,k=l;
    while(i<n1 && j<n2) a[k++] = (L[i]<=R[j])?L[i++]:R[j++];
    while(i<n1) a[k++]=L[i++];
    while(j<n2) a[k++]=R[j++];
    free(L); free(R);
}
void mergesort_rec(long *a,int l,int r){
    if (r-l<=1) return;
    int m=(l+r)/2;
    mergesort_rec(a,l,m); mergesort_rec(a,m,r); merge(a,l,m,r);
}

/* heapsort */
void heapify(long *a, int n, int i){
    int largest = i, l = 2*i+1, r = 2*i+2;
    if (l < n && a[l] > a[largest]) largest = l;
    if (r < n && a[r] > a[largest]) largest = r;
    if (largest != i){ long tmp=a[i]; a[i]=a[largest]; a[largest]=tmp; heapify(a,n,largest); }
}
void heapsort(long *a,int n){
    for (int i=n/2-1;i>=0;i--) heapify(a,n,i);
    for (int i=n-1;i>0;i--){ long tmp=a[0]; a[0]=a[i]; a[i]=tmp; heapify(a,i,0); }
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int size; MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (size<3){ if(rank==0) fprintf(stderr,"Run with -np 3\n"); MPI_Finalize(); return 1; }
    int N = 100000;
    if (argc>1) N = atoi(argv[1]);

    if (rank==0){
        long *arr = malloc(N*sizeof(long));
        srand(12345);
        for (int i=0;i<N;i++) arr[i] = rand();
        /* send size and array */
        for (int r=1;r<=2;r++){
            MPI_Send(&N,1,MPI_INT,r,TAG_ARR,MPI_COMM_WORLD);
            MPI_Send(arr,N,MPI_LONG,r,TAG_ARR,MPI_COMM_WORLD);
        }
        /* wait first DONE */
        int done;
        MPI_Status st;
        MPI_Recv(&done,1,MPI_INT,MPI_ANY_SOURCE,TAG_DONE,MPI_COMM_WORLD,&st);
        int src = st.MPI_SOURCE;
        int other = (src==1)?2:1;
        /* cancel other */
        int canc=1; MPI_Send(&canc,1,MPI_INT,other,TAG_CANCEL,MPI_COMM_WORLD);
        /* receive result size and array from winner */
        int nrecv; MPI_Recv(&nrecv,1,MPI_INT,src,TAG_RESULT,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        long *res = malloc(nrecv*sizeof(long));
        MPI_Recv(res,nrecv,MPI_LONG,src,TAG_RESULT,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("[coord] winner %d sorted first. sample res[0]=%ld\n", src, res[0]);
        free(res); free(arr);
    } else {
        int n; MPI_Recv(&n,1,MPI_INT,0,TAG_ARR,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        long *a = malloc(n*sizeof(long));
        MPI_Recv(a,n,MPI_LONG,0,TAG_ARR,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Status st; int flag=0;
        if (rank==1){
            /* MergeSort worker */
            double t0 = MPI_Wtime();
            mergesort_rec(a,0,n);
            double t1 = MPI_Wtime();
            MPI_Iprobe(0,TAG_CANCEL,MPI_COMM_WORLD,&flag,&st);
            if (!flag){
                int done=1; MPI_Send(&done,1,MPI_INT,0,TAG_DONE,MPI_COMM_WORLD);
                MPI_Send(&n,1,MPI_INT,0,TAG_RESULT,MPI_COMM_WORLD);
                MPI_Send(a,n,MPI_LONG,0,TAG_RESULT,MPI_COMM_WORLD);
                printf("[rank1] mergesort done in %g s\n", t1-t0);
            } else { int tmp; MPI_Recv(&tmp,1,MPI_INT,0,TAG_CANCEL,MPI_COMM_WORLD,&st); printf("[rank1] cancelled\n"); }
        } else if (rank==2){
            /* HeapSort worker */
            double t0 = MPI_Wtime();
            heapsort(a,n);
            double t1 = MPI_Wtime();
            MPI_Iprobe(0,TAG_CANCEL,MPI_COMM_WORLD,&flag,&st);
            if (!flag){
                int done=1; MPI_Send(&done,1,MPI_INT,0,TAG_DONE,MPI_COMM_WORLD);
                MPI_Send(&n,1,MPI_INT,0,TAG_RESULT,MPI_COMM_WORLD);
                MPI_Send(a,n,MPI_LONG,0,TAG_RESULT,MPI_COMM_WORLD);
                printf("[rank2] heapsort done in %g s\n", t1-t0);
            } else { int tmp; MPI_Recv(&tmp,1,MPI_INT,0,TAG_CANCEL,MPI_COMM_WORLD,&st); printf("[rank2] cancelled\n"); }
        }
        free(a);
    }
    MPI_Finalize();
    return 0;
}
