// subset_sum_parallel.cpp
// Compile: g++ -std=c++17 -O2 subset_sum_parallel.cpp -pthread -o subset_sum
// Usage: ./subset_sum N target num_threads
// Demo: generates random positive integers.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

int N = 30;
int num_threads = 4;
long long TARGET = 100;
vector<int> arr;
atomic<bool> found(false);
vector<int> found_subset;
mutex found_mtx;

void dfs(int idx, long long sum, vector<int>& picked) {
    if (found.load()) return;
    if (sum == TARGET) {
        lock_guard<mutex> lk(found_mtx);
        if (!found.load()) {
            found = true;
            found_subset = picked;
        }
        return;
    }
    if (idx >= N || sum > TARGET) return;
    // include
    picked.push_back(idx);
    dfs(idx+1, sum + arr[idx], picked);
    picked.pop_back();
    // exclude
    dfs(idx+1, sum, picked);
}

int main(int argc,char**argv){
    if (argc>1) N = stoi(argv[1]);
    if (argc>2) TARGET = atoll(argv[2]);
    if (argc>3) num_threads = stoi(argv[3]);

    mt19937 rng(42);
    arr.assign(N,0);
    for (int i=0;i<N;i++) arr[i] = (rng()%20) + 1;

    // Create tasks by splitting on first few items to produce >num_threads tasks
    int split = min(15, N); // number of top-level bits to expand (adjust as needed)
    vector<pair<int,long long>> tasks; // (bitmask, sum)
    tasks.emplace_back(0, 0LL); // mask over split bits: 0 means none chosen yet

    // expand tasks breadth-first until we have enough tasks or exhausted split space
    for (int b = 0; b < split && tasks.size() < (size_t)num_threads*4; ++b) {
        vector<pair<int,long long>> next;
        for (auto &t: tasks) {
            int mask = t.first; long long s = t.second;
            // exclude b
            next.emplace_back(mask, s);
            // include b
            next.emplace_back(mask | (1<<b), s + arr[b]);
        }
        tasks.swap(next);
    }

    // worker: for each task, continue DFS from index 'start'
    atomic<size_t> idx{0};
    auto worker = [&](int worker_id){
        size_t i;
        while ((i = idx.fetch_add(1)) < tasks.size() && !found.load()) {
            int mask = tasks[i].first;
            long long s = tasks[i].second;
            vector<int> picked;
            int start = 0;
            for (int b = 0; b < split; ++b) {
                if (mask & (1<<b)) { picked.push_back(b); start = b+1; }
                else start = max(start, b+1);
            }
            // set start properly as next index after highest considered bit
            start = split;
            dfs(start, s, picked);
            if (found.load()) return;
        }
    };

    vector<thread> thr;
    for (int t=0;t<num_threads;t++) thr.emplace_back(worker, t);
    for (auto &th: thr) th.join();

    if (found.load()) {
        cout << "Found subset summing to " << TARGET << " : ";
        for (int idx : found_subset) cout << arr[idx] << " ";
        cout << "\n";
    } else {
        cout << "No subset found for target " << TARGET << "\n";
    }
    return 0;
}
