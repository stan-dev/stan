#ifndef STAN__MATH__PRIM__ARR__FUN__SUB_HPP
#define STAN__MATH__PRIM__ARR__FUN__SUB_HPP

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
