#ifndef __STAN__MATH__MATRIX__SIZE_HPP__
#define __STAN__MATH__MATRIX__SIZE_HPP__

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
