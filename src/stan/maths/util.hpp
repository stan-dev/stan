#ifndef __STAN__MATHS__UTIL_HPP__
#define __STAN__MATHS__UTIL_HPP__

#include <cmath>
#include <vector>

namespace stan {

  namespace maths {

    inline double max(double a, double b) { 
      return a > b ? a : b; 
    }

    inline double min(double a, double b) { 
      return a < b ? a : b; 
    }

    // x' * y
    inline double dot(std::vector<double>& x,
                      std::vector<double>& y) {
      double sum = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        sum += x[i] * y[i];
      return sum;
    }

    // x' * x
    inline double dot_self(std::vector<double>& x) {
      double sum = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        sum += x[i] * x[i];
      return sum;
    }

    // x <- x + lambda * y
    inline void scaled_add(std::vector<double>& x, 
                           std::vector<double>& y,
                           double lambda) {
      for (size_t i = 0; i < x.size(); ++i)
        x[i] += lambda * y[i];
    }

    inline void sub(std::vector<double>& x, std::vector<double>& y,
                    std::vector<double>& result) {
      result.resize(x.size());
      for (size_t i = 0; i < x.size(); ++i)
        result[i] = x[i] - y[i];
    }

    inline double dist(const std::vector<double>& x, const std::vector<double>& y) {
      using std::sqrt;
      double result = 0;
      for (size_t i = 0; i < x.size(); ++i) {
        double diff = x[i] - y[i];
        result += diff * diff;
      }
      return sqrt(result);
    }

    inline double sum_vec(std::vector<double> x) {
      double sum = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        sum += x[i];
      return sum;
    }

    inline double max_vec(std::vector<double> x) {
      double max = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        if (x[i] > max)
          max = x[i];
      return max;
    }
      
  }
}

#endif
