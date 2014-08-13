#ifndef STAN__MATH__FUNCTIONS__MODULUS_HPP
#define STAN__MATH__FUNCTIONS__MODULUS_HPP

#include <cstddef>
#include <cstdlib>
#include <vector>

namespace stan {
  namespace math {

    inline int modulus(const int x, const int y) {
      return std::div(x, y).rem;
    }

  }
}

#endif
