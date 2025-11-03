/* spec_sim_mpi.c
   Compile: mpicc -O2 -o spec_sim_mpi spec_sim_mpi.c
   Run: mpirun -np 3 ./spec_sim_mpi STEPS
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define TAG_START 201
#define TAG_DONE 202
#define TAG_CANCEL 203
#define TAG_RESULT 204

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    int size; MPI_Comm_size(MPI_COMM_WORLD,&size);
    if (size<3){ if(rank==0) fprintf(stderr,"Run with -np 3\n"); MPI_Finalize(); return 1; }
    int STEPS = 100000000;
    if (argc>1) STEPS = atoi(argv[1]);

    if (rank==0){
        /* coordinator: send STEPS to both */
        for (int r=1;r<=2;r++) MPI_Send(&STEPS,1,MPI_INT,r,TAG_START,MPI_COMM_WORLD);
        /* wait for first DONE */
        int done; MPI_Status st;
        MPI_Recv(&done,1,MPI_INT,MPI_ANY_SOURCE,TAG_DONE,MPI_COMM_WORLD,&st);
        int src = st.MPI_SOURCE;
        int other = (src==1)?2:1;
        int canc=1; MPI_Send(&canc,1,MPI_INT,other,TAG_CANCEL,MPI_COMM_WORLD);
        /* receive result */
        long long result; MPI_Recv(&result,1,MPI_LONG_LONG,src,TAG_RESULT,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        printf("[coord] winner %d result %lld\n", src, result);
    } else if (rank==1){
        /* Conservative model: deterministic loop */
        MPI_Status st; int steps; MPI_Recv(&steps,1,MPI_INT,0,TAG_START,MPI_COMM_WORLD,&st);
        long long acc=0;
        for (int i=0;i<steps;i++){
            acc += (i % 7) - (i % 3);
            if ((i & 0xFFFF) == 0){
                int flag=0; MPI_Iprobe(0,TAG_CANCEL,MPI_COMM_WORLD,&flag,&st);
                if (flag){ int tmp; MPI_Recv(&tmp,1,MPI_INT,0,TAG_CANCEL,MPI_COMM_WORLD,&st); printf("[rank1] cancelled at i=%d\n",i); goto out1; }
            }
        }
        { int one=1; MPI_Send(&one,1,MPI_INT,0,TAG_DONE,MPI_COMM_WORLD); MPI_Send(&acc,1,MPI_LONG_LONG,0,TAG_RESULT,MPI_COMM_WORLD); printf("[rank1] done\n"); }
out1:;
    } else if (rank==2){
        /* Optimistic model: slightly different inner computation */
        MPI_Status st; int steps; MPI_Recv(&steps,1,MPI_INT,0,TAG_START,MPI_COMM_WORLD,&st);
        long long acc=0;
        for (int i=0;i<steps;i++){
            acc += ((i * 127) ^ (i << 3)) % 19;
            if ((i & 0xFFFF) == 0){
                int flag=0; MPI_Iprobe(0,TAG_CANCEL,MPI_COMM_WORLD,&flag,&st);
                if (flag){ int tmp; MPI_Recv(&tmp,1,MPI_INT,0,TAG_CANCEL,MPI_COMM_WORLD,&st); printf("[rank2] cancelled at i=%d\n",i); goto out2; }
            }
        }
        { int one=1; MPI_Send(&one,1,MPI_INT,0,TAG_DONE,MPI_COMM_WORLD); MPI_Send(&acc,1,MPI_LONG_LONG,0,TAG_RESULT,MPI_COMM_WORLD); printf("[rank2] done\n"); }
out2:;
    }
    MPI_Finalize();
    return 0;
}
