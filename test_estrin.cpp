#include <cmath>
#include <chrono>
#include <iostream>
#include <vector>
#include "lib/NumLib.h"

inline double erfc_estrin(double x) {
  if (x >= 4.0) return std::erfc(x);
  const double c[] = {
      0.99999999999999467,
      -1.1283791670934669,
      0.99999999989179178,
      -0.75225277573003257,
      0.49999997280038677,
      -0.30090091348467585,
      0.16666568909191809,
      -0.085968265180164871,
      0.041657367686491018,
      -0.019085621599354841,
      0.0083019181976808024,
      -0.0034321851709960822,
      0.0013441886912088149,
      -0.00049441318340799633,
      0.00016842398086889501,
      -5.2197190031495688e-05,
      1.4427717972747062e-05,
      -3.4847977574907564e-06,
      7.2044397335263789e-07,
      -1.2474850134635703e-07,
      1.7653207532212009e-08,
      -1.9802895244944275e-09,
      1.6886537936002582e-10,
      -1.0259331192053428e-11,
      3.9486954827591489e-13,
      -7.2263214406337637e-15,
  };
  
  double x2 = x * x;
  double x3 = x2 * x;
  double x4 = x2 * x2;
  double x8 = x4 * x4;
  double x16 = x8 * x8;

  double p0_3   = c[0] + x * c[1] + x2 * c[2] + x3 * c[3];
  double p4_7   = c[4] + x * c[5] + x2 * c[6] + x3 * c[7];
  double p8_11  = c[8] + x * c[9] + x2 * c[10] + x3 * c[11];
  double p12_15 = c[12] + x * c[13] + x2 * c[14] + x3 * c[15];
  double p16_19 = c[16] + x * c[17] + x2 * c[18] + x3 * c[19];
  double p20_23 = c[20] + x * c[21] + x2 * c[22] + x3 * c[23];
  double p24_25 = c[24] + x * c[25];

  double p0_7   = p0_3 + x4 * p4_7;
  double p8_15  = p8_11 + x4 * p12_15;
  double p16_25 = p16_19 + x4 * p20_23 + x8 * p24_25;

  double p0_15  = p0_7 + x8 * p8_15;
  double p      = p0_15 + x16 * p16_25;

  return p * std::exp(-x * x);
}

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
    
    double sum3 = 0;
    auto start3 = std::chrono::high_resolution_clock::now();
    for (double x : vals) sum3 += erfc_estrin(x);
    auto end3 = std::chrono::high_resolution_clock::now();
    
    std::cout << "erfc_cody time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end1 - start1).count() << " ms, sum=" << sum1 << "\n";
    std::cout << "std::erfc time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end2 - start2).count() << " ms, sum=" << sum2 << "\n";
    std::cout << "erfc_estrin time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end3 - start3).count() << " ms, sum=" << sum3 << "\n";
    
    return 0;
}
