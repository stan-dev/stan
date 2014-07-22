#ifndef STAN__MATH__FUNCTIONS__SUM_HPP
#define STAN__MATH__FUNCTIONS__SUM_HPP

#include <vector>
#include <cstddef>

namespace stan {
  namespace math {

    inline double sum(std::vector<double>& x) {
      double sum = x[0];
      for (size_t i = 1; i < x.size(); ++i)
        sum += x[i];
      return sum;
    }

  }
}

#endif
