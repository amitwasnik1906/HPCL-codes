/* spec_8puzzle_mpi.c
   Compile: mpicc -O2 -o spec_8puzzle_mpi spec_8puzzle_mpi.c
   Run: mpirun -np 3 ./spec_8puzzle_mpi
   Ranks:
     0 - coordinator
     1 - BFS worker
     2 - DFS worker
*/
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TAG_PUZ 11
#define TAG_DONE 12
#define TAG_CANCEL 13
#define TAG_RESULT 14

/* Represent state as 9-char string "012345678" where '0' is blank.
   Moves encoded as sequence of chars 'U','D','L','R' */
int idx_from_rc(int r,int c){ return r*3 + c; }

void print_state(const char *s){
    for (int i=0;i<9;i++){
        printf("%c", s[i]);
        if (i%3==2) printf("\n");
        else printf(" ");
    }
}

/* BFS worker */
void worker_bfs(MPI_Comm comm){
    MPI_Status st;
    char start[10];
    MPI_Recv(start,10,MPI_CHAR,0,TAG_PUZ,comm,&st);
    /* target state */
    char goal[10] = "123456780";
    /* queue */
    typedef struct { char state[10]; char path[256]; } Node;
    Node *q = malloc(100000 * sizeof(Node));
    int qh=0, qt=0;
    strcpy(q[qt].state, start);
    q[qt].path[0]=0;
    qt++;
    /* simple visited via hashing */
    int visited_size = 100003;
    char **visited = calloc(visited_size, sizeof(char*));
    while (qh < qt){
        /* check cancel */
        int flag=0; MPI_Iprobe(0,TAG_CANCEL,comm,&flag,&st);
        if (flag){
            int tmp; MPI_Recv(&tmp,1,MPI_INT,0,TAG_CANCEL,comm,&st);
            free(q); free(visited); return;
        }
        Node cur = q[qh++];
        if (strcmp(cur.state, goal)==0){
            /* send done */
            int one = 1; MPI_Send(&one,1,MPI_INT,0,TAG_DONE,comm);
            /* send result path length and path */
            int len = (int)strlen(cur.path);
            MPI_Send(&len,1,MPI_INT,0,TAG_RESULT,comm);
            MPI_Send(cur.path, len, MPI_CHAR, 0, TAG_RESULT, comm);
            free(q); free(visited); return;
        }
        /* expand neighbors */
        int p = strchr(cur.state,'0') - cur.state;
        int r = p/3, c = p%3;
        int dr[4] = {-1,1,0,0}; int dc[4]={0,0,-1,1};
        char mv[4]={'U','D','L','R'};
        for (int k=0;k<4;k++){
            int nr=r+dr[k], nc=c+dc[k];
            if (nr<0||nr>=3||nc<0||nc>=3) continue;
            char ns[10]; strcpy(ns, cur.state);
            int np = nr*3 + nc;
            ns[p] = ns[np]; ns[np] = '0';
            /* naive visited check: send to queue without dedup for brevity */
            Node next;
            strcpy(next.state, ns);
            strcpy(next.path, cur.path);
            int l = strlen(next.path); next.path[l]=mv[k]; next.path[l+1]=0;
            q[qt++] = next;
            if (qt % 100000 == 0) ; /* avoid huge memory a lot here */
        }
    }
    free(q); free(visited);
}

/* DFS worker (recursive) */
int dfs_recursive(char *state, char *path, int depth, MPI_Comm comm){
    MPI_Status st;
    int flag=0; MPI_Iprobe(0,TAG_CANCEL,comm,&flag,&st);
    if (flag){ int tmp; MPI_Recv(&tmp,1,MPI_INT,0,TAG_CANCEL,comm,&st); return 0; }
    char goal[10]="123456780";
    if (strcmp(state,goal)==0){
        int one=1; MPI_Send(&one,1,MPI_INT,0,TAG_DONE,comm);
        int len = strlen(path);
        MPI_Send(&len,1,MPI_INT,0,TAG_RESULT,comm);
        MPI_Send(path, len, MPI_CHAR, 0, TAG_RESULT, comm);
        return 1;
    }
    if (depth > 30) return 0; /* limit depth for demo */
    int p = strchr(state,'0') - state;
    int r=p/3,c=p%3;
    int dr[4] = {-1,1,0,0}; int dc[4]={0,0,-1,1};
    char mv[4]={'U','D','L','R'};
    for (int k=0;k<4;k++){
        int nr=r+dr[k], nc=c+dc[k];
        if (nr<0||nr>=3||nc<0||nc>=3) continue;
        char ns[10]; strcpy(ns, state);
        int np = nr*3 + nc;
        ns[p] = ns[np]; ns[np] = '0';
        int l = strlen(path); path[l]=mv[k]; path[l+1]=0;
        if (dfs_recursive(ns, path, depth+1, comm)) return 1;
        path[l]=0;
    }
    return 0;
}

void worker_dfs(MPI_Comm comm){
    MPI_Status st;
    char start[10];
    MPI_Recv(start,10,MPI_CHAR,0,TAG_PUZ,comm,&st);
    char path[256]="";
    dfs_recursive(start, path, 0, comm);
}

int main(int argc,char**argv){
    MPI_Init(&argc,&argv);
    int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if (rank==0){
        /* coordinator: pick a start state (hardcode small scramble) */
        char start[10] = "123405678"; /* single swap for demo */
        /* send start to both workers */
        for (int r=1;r<=2;r++) MPI_Send(start,10,MPI_CHAR,r,TAG_PUZ,MPI_COMM_WORLD);
        /* wait for first done */
        MPI_Status st; int one;
        MPI_Recv(&one,1,MPI_INT,MPI_ANY_SOURCE,TAG_DONE,MPI_COMM_WORLD,&st);
        int src = st.MPI_SOURCE;
        printf("[coord] got DONE from %d\n", src);
        int other = (src==1)?2:1;
        int canc = 1; MPI_Send(&canc,1,MPI_INT,other,TAG_CANCEL,MPI_COMM_WORLD);
        /* receive path from winner */
        int len; MPI_Recv(&len,1,MPI_INT,src,TAG_RESULT,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        char *path = malloc(len+1); MPI_Recv(path,len,MPI_CHAR,src,TAG_RESULT,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        path[len]=0;
        printf("[coord] winner %d path (%d): %s\n", src, len, path);
        free(path);
    } else if (rank==1) worker_bfs(MPI_COMM_WORLD);
    else if (rank==2) worker_dfs(MPI_COMM_WORLD);

    MPI_Finalize();
    return 0;
}
