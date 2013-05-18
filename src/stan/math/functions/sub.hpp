#ifndef __STAN__MATH__FUNCTIONS__SUB_HPP__
#define __STAN__MATH__FUNCTIONS__SUB_HPP__

#include <vector>
#include <cstddef>

namespace stan {
  namespace math {

    inline void sub(std::vector<double>& x, std::vector<double>& y,
                    std::vector<double>& result) {
      result.resize(x.size());
      for (size_t i = 0; i < x.size(); ++i)
        result[i] = x[i] - y[i];
    }

  }
}

#endif
