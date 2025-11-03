// tsp_parallel.cpp
// Compile: g++ -std=c++17 -O2 tsp_parallel.cpp -pthread -o tsp_parallel
// Usage: ./tsp_parallel N num_threads
// Example generates random symmetric distance matrix for demo.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

int N = 10;
int num_threads = 4;
vector<vector<int>> distmat;
atomic<int> best_cost;
vector<int> best_tour;
mutex best_mtx;

void dfs(int cur, vector<int>& tour, vector<char>& used, int cost) {
    if (cost >= best_cost.load()) return; // prune
    if ((int)tour.size() == N) {
        cost += distmat[cur][0];
        if (cost < best_cost.load()) {
            lock_guard<mutex> lk(best_mtx);
            if (cost < best_cost.load()) {
                best_cost.store(cost);
                best_tour = tour;
            }
        }
        return;
    }
    for (int nxt = 1; nxt < N; ++nxt) {
        if (!used[nxt]) {
            used[nxt] = 1;
            tour.push_back(nxt);
            dfs(nxt, tour, used, cost + distmat[cur][nxt]);
            tour.pop_back();
            used[nxt] = 0;
        }
    }
}

int main(int argc,char**argv){
    if (argc>1) N = stoi(argv[1]);
    if (argc>2) num_threads = stoi(argv[2]);
    // generate random symmetric distances
    mt19937 rng(12345);
    distmat.assign(N, vector<int>(N, 0));
    for (int i=0;i<N;i++) for (int j=i+1;j<N;j++){
        int d = rng()%100 + 1;
        distmat[i][j] = distmat[j][i] = d;
    }

    best_cost.store(INT_MAX);
    // Prepare tasks: choose second city after start (0) -> many tasks for threads
    vector<int> initial_choices;
    for (int c=1;c<N;c++) initial_choices.push_back(c);

    atomic<size_t> idx{0};
    auto worker = [&](){
        size_t i;
        while ((i = idx.fetch_add(1)) < initial_choices.size()) {
            int second = initial_choices[i];
            vector<int> tour;
            tour.push_back(0);
            tour.push_back(second);
            vector<char> used(N,0);
            used[0]=1; used[second]=1;
            dfs(second, tour, used, distmat[0][second]);
        }
    };

    vector<thread> thr;
    int launch = min((int)initial_choices.size(), num_threads);
    auto tstart = chrono::steady_clock::now();
    for (int t=0;t<launch;t++) thr.emplace_back(worker);
    for (auto &th: thr) th.join();
    auto tend = chrono::steady_clock::now();
    double sec = chrono::duration<double>(tend - tstart).count();

    cout << "N="<<N<<" threads="<<launch<<" best_cost="<<best_cost.load()<<" time="<<sec<<"s\n";
    if (!best_tour.empty()){
        cout<<"tour: 0 ";
        for(auto v: best_tour) cout<<v<<" ";
        cout<<0<<"\n";
    }
    return 0;
}
