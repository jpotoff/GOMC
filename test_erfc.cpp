#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include "lib/NumLib.h"

int main() {
    std::vector<double> vals(10000000);
    for (size_t i = 0; i < vals.size(); ++i) {
        vals[i] = 4.0 * i / vals.size();
    }
    
    double sum1 = 0;
    auto start1 = std::chrono::high_resolution_clock::now();
    for (double x : vals) sum1 += num::erfc_cody(x);
    auto end1 = std::chrono::high_resolution_clock::now();
    
    double sum2 = 0;
    auto start2 = std::chrono::high_resolution_clock::now();
    for (double x : vals) sum2 += std::erfc(x);
    auto end2 = std::chrono::high_resolution_clock::now();
    
    std::cout << "erfc_cody time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms, sum=" << sum1 << "\n";
    std::cout << "std::erfc time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << " ms, sum=" << sum2 << "\n";
    
    return 0;
}
