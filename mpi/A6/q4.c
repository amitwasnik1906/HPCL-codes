#include <mpi.h>
#include <stdio.h>

int main(int argc, char* argv[]) {
    int rank, size, send_val, recv_val;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    send_val = rank;

    // Send to right neighbor, receive from left
    MPI_Sendrecv(&send_val, 1, MPI_INT, right, 0,
                 &recv_val, 1, MPI_INT, left, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("Process %d sent %d to %d and received %d from %d\n",
           rank, send_val, right, recv_val, left);

    MPI_Finalize();
    return 0;
}
