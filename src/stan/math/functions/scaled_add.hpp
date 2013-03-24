#ifndef __STAN__MATH__FUNCTIONS__SCALED_ADD_HPP__
#define __STAN__MATH__FUNCTIONS__SCALED_ADD_HPP__

#include <vector>

namespace stan {
  namespace math {

    // x <- x + lambda * y
    inline void scaled_add(const std::vector<double>& x, 
                           const std::vector<double>& y,
                           const double lambda) {
      for (size_t i = 0; i < x.size(); ++i)
        x[i] += lambda * y[i];
    }

  }
}

#endif
