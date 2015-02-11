#ifndef STAN__MATH__MATRIX__SIZE_HPP
#define STAN__MATH__MATRIX__SIZE_HPP

#include <vector>

namespace stan {
  namespace math {

    template <typename T>
    inline int
    size(const std::vector<T>& x) {
      return x.size();
    }

  }
}
#endif
