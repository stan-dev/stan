#ifndef __STAN__MATH__FUNCTIONS__DOT_HPP__
#define __STAN__MATH__FUNCTIONS__DOT_HPP__

#include <vector>
#include <cstddef>

namespace stan {
  namespace math {

    // x' * y
    inline double dot(const std::vector<double>& x,
                      const std::vector<double>& y) {
      double sum = 0.0;
      for (size_t i = 0; i < x.size(); ++i)
        sum += x[i] * y[i];
      return sum;
    }

  }
}

#endif
