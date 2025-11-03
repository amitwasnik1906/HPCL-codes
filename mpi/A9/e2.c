// maze_bfs_parallel.cpp
// Compile: g++ -std=c++17 -O2 maze_bfs_parallel.cpp -pthread -o maze_bfs
// Usage: ./maze_bfs rows cols num_threads
// Program randomly generates a maze with open cells (for demonstration).
#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
using namespace std;

struct Pos{int r,c; int d;};

int R=50, C=50;
vector<char> grid;
vector<char> visited;
mutex vis_mtx;
deque<Pos> q;
mutex q_mtx;
condition_variable q_cv;
atomic<bool> found(false);
Pos target{-1,-1,0};
vector<int> parent; // flatten index -> prev index, -1 for none

int idx(int r,int c){return r*C + c;}

bool inb(int r,int c){return r>=0 && r<R && c>=0 && c<C;}

void worker_thread() {
    while (!found.load()) {
        Pos cur;
        {
            unique_lock<mutex> lk(q_mtx);
            if (q.empty()) {
                // wait briefly then exit if still empty
                if(q_cv.wait_for(lk, chrono::milliseconds(10)) == cv_status::timeout) {
                    if (q.empty()) return;
                }
            }
            if (q.empty()) continue;
            cur = q.front(); q.pop_front();
        }
        int r = cur.r, c = cur.c;
        int dirs[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
        for (auto &d:dirs) {
            int nr=r+d[0], nc=c+d[1];
            if(!inb(nr,nc)) continue;
            int id = idx(nr,nc);
            // try to mark visited
            bool do_push = false;
            {
                lock_guard<mutex> lk(vis_mtx);
                if(!visited[id] && grid[id] == '.') {
                    visited[id]=1;
                    parent[id]= idx(r,c);
                    do_push = true;
                }
            }
            if(do_push) {
                Pos np{nr,nc,cur.d+1};
                {
                    lock_guard<mutex> lk(q_mtx);
                    q.push_back(np);
                }
                q_cv.notify_all();
                if(nr==target.r && nc==target.c) {
                    found.store(true);
                    q_cv.notify_all();
                    return;
                }
            }
        }
    }
}

int main(int argc,char**argv){
    int threads = 4;
    if(argc>=2) R = atoi(argv[1]);
    if(argc>=3) C = atoi(argv[2]);
    if(argc>=4) threads = atoi(argv[3]);
    grid.assign(R*C,'.');
    visited.assign(R*C,0);
    parent.assign(R*C,-1);

    // random obstacles
    mt19937_64 rng(123456);
    for(int i=0;i<R;i++) for(int j=0;j<C;j++){
        if(rng()%100 < 30) grid[idx(i,j)] = '#'; // 30% walls
    }
    // ensure start and target free
    grid[idx(0,0)]='.';
    target = {R-1,C-1,0};
    grid[idx(target.r,target.c)] = '.';

    // init BFS
    {
        lock_guard<mutex> lk(q_mtx);
        visited[idx(0,0)] = 1;
        parent[idx(0,0)] = -1;
        q.push_back({0,0,0});
    }

    vector<thread> thr;
    for(int t=0;t<threads;t++) thr.emplace_back(worker_thread);
    for(auto &th:thr) th.join();

    if(found.load()){
        // reconstruct path length
        int cur = idx(target.r,target.c);
        int steps = 0;
        while(parent[cur] != -1) { cur = parent[cur]; steps++; }
        cout << "Path found of length " << steps << "\n";
    } else cout << "No path found\n";

    return 0;
}
