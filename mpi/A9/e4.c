// graph_coloring_parallel.cpp
// Compile: g++ -std=c++17 -O2 graph_coloring_parallel.cpp -pthread -o gcolor
// Usage: ./gcolor num_vertices K num_threads
// Example builds a sample graph (triangle + extra) for demo.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

atomic<bool> found_color(false);
vector<int> solution;

bool valid_color(const vector<vector<int>>& adj, const vector<int>& color, int v, int c) {
    for (int u : adj[v]) if (color[u] == c) return false;
    return true;
}

bool dfs_color(const vector<vector<int>>& adj, vector<int>& color, int v, int K) {
    if (found_color.load()) return true;
    int n = adj.size();
    if (v == n) {
        solution = color;
        found_color.store(true);
        return true;
    }
    for (int c = 0; c < K; ++c) {
        if (valid_color(adj, color, v, c)) {
            color[v] = c;
            if (dfs_color(adj, color, v+1, K)) return true;
            color[v] = -1;
        }
    }
    return false;
}

int main(int argc,char**argv){
    int n=8;
    int K=3;
    int num_threads=4;
    if (argc>1) n=atoi(argv[1]);
    if (argc>2) K=atoi(argv[2]);
    if (argc>3) num_threads=atoi(argv[3]);

    // Build demo graph: simple random-ish graph for testing
    vector<vector<int>> adj(n);
    // make a graph: connect i to i+1 and some random edges
    for (int i=0;i<n-1;i++){ adj[i].push_back(i+1); adj[i+1].push_back(i); }
    for (int i=0;i<n;i++){
        if (i+3 < n) { adj[i].push_back(i+3); adj[i+3].push_back(i); }
    }
    // Add a triangle to make coloring interesting
    if (n>=3) { adj[0].push_back(1); adj[1].push_back(0); adj[1].push_back(2); adj[2].push_back(1); adj[2].push_back(0); adj[0].push_back(2); }

    // We'll assign colors for vertex 0 across threads
    vector<int> initial_color(n, -1);
    vector<int> initial_colors;
    for (int c = 0; c < K; ++c) initial_colors.push_back(c);

    vector<thread> threads;
    for (int c : initial_colors) {
        threads.emplace_back([&,c](){
            vector<int> color = initial_color;
            color[0] = c;
            dfs_color(adj, color, 1, K);
        });
        if ((int)threads.size() >= num_threads) break;
    }
    for (auto &t: threads) if (t.joinable()) t.join();

    if (found_color.load()){
        cout << "Found coloring with K="<<K<<":\n";
        for (int i=0;i<n;i++) cout << solution[i] << (i+1==n?'\n':' ');
    } else cout << "No coloring with K="<<K<<"\n";
    return 0;
}
