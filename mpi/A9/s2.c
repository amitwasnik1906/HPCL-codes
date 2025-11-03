// spec_quicksort.cpp
// Compile: g++ -std=c++17 -O2 spec_quicksort.cpp -pthread -o spec_quicksort
// Usage: ./spec_quicksort N num_threads
// N = array size, num_threads used for top-level speculative tasks
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

using ll = long long;

mt19937_64 rng(123456);

int partition_once(vector<int>& a, int l, int r, int pivot, vector<int>& out_left, vector<int>& out_right){
    out_left.clear(); out_right.clear();
    for (int i = l; i < r; ++i){
        if (a[i] < pivot) out_left.push_back(a[i]);
        else out_right.push_back(a[i]);
    }
    return (int)out_left.size();
}

// non-speculative quicksort for small arrays
void seq_quicksort(vector<int>& a, int l, int r){
    if (r - l <= 32) { sort(a.begin()+l, a.begin()+r); return; }
    int pivot = a[l + (rng() % (r-l))];
    int i = l, j = r-1;
    // Lomuto-like partition for simplicity (in-place)
    vector<int> lessv, greatv;
    partition_once(a, l, r, pivot, lessv, greatv);
    int idx = l;
    for (int v: lessv) a[idx++] = v;
    for (int v: greatv) a[idx++] = v;
    // split point
    int mid = l + (int)lessv.size();
    seq_quicksort(a, l, mid);
    seq_quicksort(a, mid, r);
}

// speculative quicksort: at each node try two pivots in parallel
void spec_quicksort(vector<int>& a, int l, int r, int depth=0){
    if (r - l <= 1024) { seq_quicksort(a,l,r); return; }
    // choose two candidate pivots
    int p1 = a[l + (rng() % (r-l))];
    int p2 = a[l + (rng() % (r-l))];
    if (p1 == p2) p2 = p1 + 1;

    vector<int> left1, right1, left2, right2;
    thread th1([&]{ partition_once(a, l, r, p1, left1, right1); });
    thread th2([&]{ partition_once(a, l, r, p2, left2, right2); });
    th1.join(); th2.join();

    // heuristic: choose pivot with most balanced partition
    int bal1 = abs((int)left1.size() - (int)right1.size());
    int bal2 = abs((int)left2.size() - (int)right2.size());
    vector<int> chosen_left, chosen_right;
    if (bal1 <= bal2) { chosen_left.swap(left1); chosen_right.swap(right1); }
    else { chosen_left.swap(left2); chosen_right.swap(right2); }

    // write back to array
    int idx=l;
    for (int v: chosen_left) a[idx++]=v;
    for (int v: chosen_right) a[idx++]=v;
    int mid = l + (int)chosen_left.size();

    // recurse
    spec_quicksort(a, l, mid, depth+1);
    spec_quicksort(a, mid, r, depth+1);
}

int main(int argc,char**argv){
    int N = 1<<20;
    int threads = 4;
    if(argc>1) N = atoi(argv[1]);
    if(argc>2) threads = atoi(argv[2]);

    vector<int> a(N);
    for (int i=0;i<N;i++) a[i] = rng() & 0x7fffffff;

    auto b = a;
    auto t0 = chrono::steady_clock::now();
    spec_quicksort(a, 0, N);
    auto t1 = chrono::steady_clock::now();
    double tspec = chrono::duration<double>(t1-t0).count();

    auto t2 = chrono::steady_clock::now();
    sort(b.begin(), b.end());
    auto t3 = chrono::steady_clock::now();
    double tseq = chrono::duration<double>(t3-t2).count();

    // verify
    cout << "N="<<N<<" spec_time="<<tspec<<" seq_sort_time="<<tseq<<"\n";
    if (a==b) cout<<"Correctly sorted\n"; else cout<<"Mismatch!\n";
    return 0;
}
