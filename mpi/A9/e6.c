// word_search_parallel.cpp
// Compile: g++ -std=c++17 -O2 word_search_parallel.cpp -pthread -o word_search
// Usage: ./word_search R C num_threads
// Demo builds a random grid and searches for a few sample words.
#include <bits/stdc++.h>
#include <thread>
#include <mutex>
using namespace std;

int R=20, C=30, threads_n=4;
vector<string> grid;
vector<string> words = {"THIS","SEARCH","WORD","TEST","GRID"};
mutex out_mtx;
vector<tuple<string,int,int,int,int>> results; // word, r, c, dr, dc

bool inb(int r,int c){ return r>=0 && r<R && c>=0 && c<C; }

bool match_from(int r,int c, int dr, int dc, const string &w){
    for (int k=0;k<(int)w.size();k++){
        int nr = r + k*dr, nc = c + k*dc;
        if (!inb(nr,nc) || grid[nr][nc] != w[k]) return false;
    }
    return true;
}

void worker(int start_row, int end_row){
    int dirs[8][2] = {{-1,0},{1,0},{0,-1},{0,1},{-1,-1},{-1,1},{1,-1},{1,1}};
    for (int r = start_row; r < end_row; ++r){
        for (int c = 0; c < C; ++c){
            for (auto &w: words){
                if (grid[r][c] != w[0]) continue;
                for (auto &d: dirs){
                    if (match_from(r,c,d[0],d[1],w)){
                        lock_guard<mutex> lk(out_mtx);
                        results.emplace_back(w, r, c, d[0], d[1]);
                    }
                }
            }
        }
    }
}

int main(int argc,char**argv){
    if(argc>1) R=stoi(argv[1]);
    if(argc>2) C=stoi(argv[2]);
    if(argc>3) threads_n=stoi(argv[3]);

    // random grid
    mt19937 rng(1234);
    grid.assign(R, string(C,'A'));
    for (int i=0;i<R;i++) for (int j=0;j<C;j++) grid[i][j] = 'A' + (rng()%26);

    // inject some words deliberately to test
    if (R>2 && C>10){
        grid[2].replace(2,4,"THIS");
        // horizontal SEARCH
        if (C>10) grid[5].replace(3,6,"SEARCH");
    }

    vector<thread> thr;
    int block = (R + threads_n -1)/threads_n;
    for (int t=0;t<threads_n;t++){
        int sr = t*block;
        int er = min(R, sr + block);
        if (sr >= er) continue;
        thr.emplace_back(worker, sr, er);
    }
    for (auto &th: thr) th.join();

    cout << "Found " << results.size() << " matches\n";
    for (auto &tp: results){
        string w; int r,c,dr,dc;
        tie(w,r,c,dr,dc) = tp;
        cout<<w<<" at ("<<r<<","<<c<<") dir=("<<dr<<","<<dc<<")\n";
    }
    return 0;
}
