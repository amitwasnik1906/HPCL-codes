// spec_pathfind.cpp
// Compile: g++ -std=c++17 -O2 spec_pathfind.cpp -pthread -o spec_pathfind
// Usage: ./spec_pathfind rows cols
// Uses 4-neighbor grid, unit weights. A* uses Manhattan heuristic.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

struct Node { int r,c; double g,f; bool operator<(Node const& o) const { return f > o.f; } };

int R=100, C=100;
atomic<bool> done(false);
vector<int> parent; // parent index or -1
mutex sol_mtx;
vector<int> solution_parents;

int idx(int r,int c){return r*C + c;}
bool inb(int r,int c){return r>=0 && r<R && c>=0 && c<C;}

vector<int> reconstruct(int s, int t){
    vector<int> path;
    int cur = t;
    while (cur != -1) { path.push_back(cur); if (cur==s) break; cur = parent[cur]; }
    reverse(path.begin(), path.end());
    return path;
}

void run_dijkstra(pair<int,int> start, pair<int,int> target){
    int S = idx(start.first,start.second);
    int T = idx(target.first,target.second);
    int N = R*C;
    vector<double> dist(N, 1e100);
    vector<int> par(N,-1);
    using P = pair<double,int>;
    priority_queue<P, vector<P>, greater<P>> pq;
    dist[S]=0; pq.push({0,S});
    int dirs[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
    while(!pq.empty() && !done.load()){
        auto [d,u] = pq.top(); pq.pop();
        if (d != dist[u]) continue;
        if (u==T) {
            lock_guard<mutex> lk(sol_mtx);
            parent = par;
            done.store(true);
            return;
        }
        int ur=u/C, uc=u%C;
        for(auto &di:dirs){
            int vr=ur+di[0], vc=uc+di[1];
            if(!inb(vr,vc)) continue;
            int v = idx(vr,vc);
            double nd = d + 1.0;
            if (nd < dist[v]) { dist[v]=nd; par[v]=u; pq.push({nd,v}); }
        }
    }
}

void run_astar(pair<int,int> start, pair<int,int> target){
    int S = idx(start.first,start.second);
    int T = idx(target.first,target.second);
    int N = R*C;
    vector<double> gscore(N, 1e100);
    vector<int> par(N,-1);
    auto h = [&](int v){ int vr=v/C, vc=v%C; int tr=target.first, tc=target.second; return (double)(abs(vr-tr)+abs(vc-tc)); };
    priority_queue<Node> pq;
    gscore[S]=0; pq.push({start.first,start.second,0,h(S)});
    int dirs[4][2]={{-1,0},{1,0},{0,-1},{0,1}};
    while(!pq.empty() && !done.load()){
        auto node = pq.top(); pq.pop();
        int u = idx(node.r,node.c);
        if (u==T) {
            lock_guard<mutex> lk(sol_mtx);
            parent = par;
            done.store(true);
            return;
        }
        for(auto &di:dirs){
            int vr=node.r+di[0], vc=node.c+di[1];
            if(!inb(vr,vc)) continue;
            int v = idx(vr,vc);
            double tentative = gscore[u] + 1.0;
            if (tentative < gscore[v]) {
                gscore[v]=tentative; par[v]=u;
                pq.push({vr,vc, tentative, tentative + h(v)});
            }
        }
    }
}

int main(int argc,char**argv){
    if(argc>1) R = atoi(argv[1]);
    if(argc>2) C = atoi(argv[2]);
    pair<int,int> S = {0,0}, T = {R-1,C-1};

    // Run both algorithms speculatively
    auto t0 = chrono::steady_clock::now();
    thread t1(run_dijkstra, S, T);
    thread t2(run_astar, S, T);
    t1.join(); t2.join();
    auto t1t = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(t1t-t0).count();

    if (done.load()) {
        int Tidx = idx(T.first,T.second);
        // reconstruct path length
        int cur = Tidx; int len=0;
        while (cur != -1 && cur != idx(S.first,S.second)) { cur = parent[cur]; ++len; if (len > R*C) break; }
        cout << "Found path length ~" << len << " time="<<elapsed<<"s\n";
    } else cout << "No path found\n";

    return 0;
}
