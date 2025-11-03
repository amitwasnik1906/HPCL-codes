// spec_ifelse.cpp
// Compile: g++ -std=c++17 -O2 spec_ifelse.cpp -pthread -o spec_ifelse
// Usage: ./spec_ifelse <x>
// Demonstrates speculative evaluation of two branches.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

int main(int argc,char**argv){
    double x = -2.5;
    if(argc>1) x = atof(argv[1]);

    atomic<bool> ready1(false), ready2(false);
    double res_sqrt=0.0, res_log=0.0;

    // expensive dummy function to simulate heavy work
    auto heavy_sqrt = [&](double v){
        // simulate work
        double s=0;
        for(int i=0;i<2000000;i++) s += sin(v + i*1e-6);
        if (v >= 0) res_sqrt = sqrt(v); else res_sqrt = NAN;
        ready1.store(true);
    };
    auto heavy_log = [&](double v){
        double s=0;
        for(int i=0;i<2000000;i++) s += cos(v + i*1e-6);
        res_log = log(fabs(v)+1e-12);
        ready2.store(true);
    };

    auto t0 = chrono::steady_clock::now();
    thread t1(heavy_sqrt, x);
    thread t2(heavy_log, x);

    // condition resolves here (in real systems later)
    bool cond = (x > 0);

    // wait for both tasks to finish (we computed both speculatively)
    t1.join(); t2.join();
    auto t1t = chrono::steady_clock::now();
    double elapsed = chrono::duration<double>(t1t - t0).count();

    double final_res = cond ? res_sqrt : res_log;
    cout << fixed << setprecision(8);
    cout << "x="<<x<<" cond(x>0)="<<cond<<"\n";
    cout << "sqrt branch result (computed) = "<<res_sqrt<<"\n";
    cout << "log branch result (computed)  = "<<res_log<<"\n";
    cout << "Selected final result = "<<final_res<<"\n";
    cout << "Total speculative time (both computed) = "<<elapsed<<" s\n";
    return 0;
}
