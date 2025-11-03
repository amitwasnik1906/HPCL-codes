// sudoku_parallel.cpp
// Compile: g++ -std=c++17 -O2 sudoku_parallel.cpp -pthread -o sudoku
// Usage: ./sudoku
// The program contains a sample puzzle (0 denotes empty). It spawns threads per candidate for first empty cell.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

atomic<bool> solved(false);
mutex sol_mtx;
array<array<int,9>,9> solution_board;

bool valid(const array<array<int,9>,9>& board, int r, int c, int v) {
    for (int i = 0; i < 9; ++i) if (board[r][i] == v) return false;
    for (int i = 0; i < 9; ++i) if (board[i][c] == v) return false;
    int br = (r/3)*3, bc = (c/3)*3;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) if (board[br+i][bc+j] == v) return false;
    return true;
}

bool solve_dfs(array<array<int,9>,9>& board) {
    if (solved.load()) return true;
    int mr=-1, mc=-1;
    for (int i=0;i<9;i++) for (int j=0;j<9;j++) if (board[i][j]==0) { mr=i; mc=j; goto found; }
found:
    if (mr==-1) {
        // solved
        {
            lock_guard<mutex> lk(sol_mtx);
            if (!solved.load()) solution_board = board;
            solved.store(true);
        }
        return true;
    }
    for (int v=1; v<=9; ++v) {
        if (valid(board,mr,mc,v)) {
            board[mr][mc] = v;
            if (solve_dfs(board)) return true;
            board[mr][mc] = 0;
        }
    }
    return false;
}

int main(){
    // sample puzzle (0 = empty)
    array<array<int,9>,9> board = {{
        {{5,3,0,0,7,0,0,0,0}},
        {{6,0,0,1,9,5,0,0,0}},
        {{0,9,8,0,0,0,0,6,0}},
        {{8,0,0,0,6,0,0,0,3}},
        {{4,0,0,8,0,3,0,0,1}},
        {{7,0,0,0,2,0,0,0,6}},
        {{0,6,0,0,0,0,2,8,0}},
        {{0,0,0,4,1,9,0,0,5}},
        {{0,0,0,0,8,0,0,7,9}}
    }};

    // find first empty
    int fr=-1, fc=-1;
    for (int i=0;i<9;i++) for (int j=0;j<9;j++) if (board[i][j]==0) { fr=i; fc=j; goto fx; }
fx:
    if (fr==-1) { cout << "Already solved\n"; return 0; }

    vector<int> candidates;
    for (int v=1; v<=9; ++v) if (valid(board,fr,fc,v)) candidates.push_back(v);
    vector<thread> threads;
    for (int x : candidates) {
        threads.emplace_back([board,fr,fc,x](){ 
            auto b = board;
            b[fr][fc] = x;
            solve_dfs(b);
        });
    }
    for (auto &t: threads) if (t.joinable()) t.join();

    if (solved.load()){
        cout << "Solution found:\n";
        for (int i=0;i<9;i++){
            for (int j=0;j<9;j++) cout << solution_board[i][j] << ' ';
            cout << '\n';
        }
    } else cout << "No solution\n";
    return 0;
}
