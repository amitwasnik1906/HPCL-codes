// eight_puzzle_parallel.cpp
// Compile: g++ -std=c++17 -O2 eight_puzzle_parallel.cpp -pthread -o eight_puzzle
// Usage: ./eight_puzzle num_threads
// Demo uses a small scramble from the goal state.
#include <bits/stdc++.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
using namespace std;

using State = string; // 9 chars, '0' is blank
atomic<bool> found(false);
mutex visited_mtx;
unordered_set<State> visited;
deque<pair<State,int>> q; // state, depth
mutex q_mtx;
condition_variable q_cv;
State goal = "123456780";
vector<State> parents; // optional not used for full path in this simple demo
int R = 3, C = 3;

vector<State> neighbors(const State &s) {
    vector<State> out;
    int pos = s.find('0');
    int r = pos / C, c = pos % C;
    int dirs[4][2] = {{-1,0},{1,0},{0,-1},{0,1}};
    for (auto &d: dirs) {
        int nr = r + d[0], nc = c + d[1];
        if (nr>=0 && nr<R && nc>=0 && nc<C) {
            State t = s;
            swap(t[pos], t[nr*C + nc]);
            out.push_back(t);
        }
    }
    return out;
}

void worker() {
    while (!found.load()) {
        pair<State,int> cur;
        {
            unique_lock<mutex> lk(q_mtx);
            if (q.empty()) {
                if (q_cv.wait_for(lk, chrono::milliseconds(20)) == cv_status::timeout) {
                    if (q.empty()) return;
                }
            }
            if (q.empty()) continue;
            cur = q.front(); q.pop_front();
        }
        State s = cur.first;
        int d = cur.second;
        auto nb = neighbors(s);
        for (auto &t: nb) {
            bool push = false;
            {
                lock_guard<mutex> lk(visited_mtx);
                if (!visited.count(t)) {
                    visited.insert(t);
                    push = true;
                }
            }
            if (push) {
                if (t == goal) {
                    found.store(true);
                    cout << "Found goal at depth " << d+1 << "\n";
                    q_cv.notify_all();
                    return;
                }
                {
                    lock_guard<mutex> lk(q_mtx);
                    q.emplace_back(t, d+1);
                }
                q_cv.notify_all();
            }
        }
    }
}

int main(int argc,char**argv){
    int threads = 4;
    if (argc>1) threads = stoi(argv[1]);

    // start from a small scramble
    State start = "123450678"; // blank swapped -> 1 move away
    // you can set a harder start, but BFS memory grows fast

    {
        lock_guard<mutex> lk(visited_mtx);
        visited.insert(start);
    }
    {
        lock_guard<mutex> lk(q_mtx);
        q.emplace_back(start, 0);
    }

    vector<thread> thr;
    auto t0 = chrono::steady_clock::now();
    for (int i=0;i<threads;i++) thr.emplace_back(worker);
    for (auto &t: thr) t.join();
    auto t1 = chrono::steady_clock::now();
    double sec = chrono::duration<double>(t1-t0).count();
    cout << "Finished. threads="<<threads<<" time="<<sec<<"s visited="<<visited.size()<<"\n";
    return 0;
}
