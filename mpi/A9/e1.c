// nqueens_parallel.cpp
// Compile: g++ -std=c++17 -O2 nqueens_parallel.cpp -pthread -o nqueens
// Usage: ./nqueens N num_threads
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

int N;
atomic<long long> total_solutions{0};

bool safe(const vector<int>& cols, int row, int c) {
    for (int r = 0; r < row; ++r) {
        int cc = cols[r];
        if (cc == c) return false;
        if (abs(cc - c) == abs(r - row)) return false;
    }
    return true;
}

void solve_from_prefix(vector<int> cols, int row) {
    if (row == N) {
        total_solutions.fetch_add(1, memory_order_relaxed);
        return;
    }
    for (int c = 0; c < N; ++c) {
        if (safe(cols, row, c)) {
            cols[row] = c;
            solve_from_prefix(cols, row + 1);
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " N num_threads\n";
        return 1;
    }
    N = stoi(argv[1]);
    int num_threads = stoi(argv[2]);
    if (num_threads < 1) num_threads = 1;

    vector<thread> threads;
    // we will assign distinct first-row column positions to threads.
    // If threads > N, assign combinations of first two rows (simple extension).
    vector<pair<int,int>> prefixes; // pair(row0col, row1col or -1)
    for (int c0 = 0; c0 < N; ++c0) {
        prefixes.emplace_back(c0, -1);
    }
    if (num_threads > N) {
        // use two-row prefixes to get more tasks
        prefixes.clear();
        for (int c0 = 0; c0 < N; ++c0) {
            for (int c1 = 0; c1 < N; ++c1) {
                if (c1 == c0 || abs(c1 - c0) == 1) continue; // quick prune
                prefixes.emplace_back(c0, c1);
            }
        }
    }

    atomic<size_t> idx{0};
    auto worker = [&](){
        size_t i;
        while ((i = idx.fetch_add(1, memory_order_relaxed)) < prefixes.size()) {
            auto p = prefixes[i];
            vector<int> cols(N, -1);
            cols[0] = p.first;
            int start_row = 1;
            if (p.second != -1) {
                cols[1] = p.second;
                start_row = 2;
            }
            // validate prefix
            bool ok = true;
            for (int r = 0; r < start_row && ok; ++r)
                for (int s = r+1; s < start_row && ok; ++s)
                    if (cols[r] == cols[s] || abs(cols[r] - cols[s]) == abs(r - s)) ok = false;
            if (!ok) continue;
            solve_from_prefix(cols, start_row);
        }
    };

    int launch = min((int)prefixes.size(), num_threads);
    for (int t = 0; t < launch; ++t) threads.emplace_back(worker);
    for (auto &th : threads) th.join();

    cout << "N=" << N << " solutions=" << total_solutions.load() << " (threads=" << launch << ")\n";
    return 0;
}
