#ifndef STAN__MATH__FUNCTIONS__MODULUS_HPP
#define STAN__MATH__FUNCTIONS__MODULUS_HPP

#include <vector>
#include <cstddef>
#include <cstdlib>

namespace stan {
  namespace math {

    inline int modulus(const int x, const int y) {
      return std::div(x, y).rem;
    }

  }
}

#endif
