#ifndef STAN__MATH__FUNCTIONS__DIVIDE_HPP
#define STAN__MATH__FUNCTIONS__DIVIDE_HPP

#include <vector>
#include <cstddef>
#include <cstdlib>

namespace stan {
  namespace math {

    inline int divide(const int x, const int y) {
      return std::div(x, y).quot;
    }

  }
}

#endif
