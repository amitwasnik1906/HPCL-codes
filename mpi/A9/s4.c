// spec_poly.cpp
// Compile: g++ -std=c++17 -O2 spec_poly.cpp -pthread -o spec_poly
// Usage: ./spec_poly degree x
// Demo compares Horner and naive power-sum methods.
#include <bits/stdc++.h>
#include <thread>
#include <atomic>
using namespace std;

int main(int argc,char**argv){
    int deg = 500000;
    double x = 1.000001;
    if(argc>1) deg = atoi(argv[1]);
    if(argc>2) x = atof(argv[2]);

    // construct polynomial coefficients (random small numbers)
    vector<double> a(deg+1);
    mt19937_64 rng(1234);
    for (int i=0;i<=deg;i++) a[i] = (double)((rng()%1000)-500)/1000.0;

    atomic<bool> done(false);
    double res_horner=0.0, res_naive=0.0;
    double t_horner=0.0, t_naive=0.0;

    auto horner = [&](){
        auto t0 = chrono::steady_clock::now();
        double r = a[deg];
        for (int i=deg-1;i>=0;i--) r = r * x + a[i];
        auto t1 = chrono::steady_clock::now();
        res_horner = r;
        t_horner = chrono::duration<double>(t1-t0).count();
        done.store(true);
    };

    auto naive = [&](){
        auto t0 = chrono::steady_clock::now();
        double r = 0;
        double xp = 1;
        for (int i=0;i<=deg;i++){
            r += a[i] * xp;
            xp *= x;
        }
        auto t1 = chrono::steady_clock::now();
        res_naive = r;
        t_naive = chrono::duration<double>(t1-t0).count();
        done.store(true);
    };

    thread th1(horner);
    thread th2(naive);
    th1.join(); th2.join();

    cout<<fixed<<setprecision(12);
    cout<<"Horner: time="<<t_horner<<" res="<<res_horner<<"\n";
    cout<<"Naive : time="<<t_naive<<" res="<<res_naive<<"\n";
    double diff = fabs(res_horner - res_naive);
    cout<<"abs diff = "<<diff<<"\n";
    double eps = 1e-8;
    if (diff < eps) {
        cout<<"Results similar. Selecting faster method: ";
        if (t_horner <= t_naive) cout<<"Horner\n"; else cout<<"Naive\n";
    } else {
        cout<<"Results differ. Selecting Horner for numerical stability.\n";
    }
    return 0;
}
