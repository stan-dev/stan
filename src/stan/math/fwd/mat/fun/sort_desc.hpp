#ifndef STAN_MATH_FWD_MAT_FUN_SORT_DESC_HPP
#define STAN_MATH_FWD_MAT_FUN_SORT_DESC_HPP

#include <stan/math/fwd/core.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>    // std::sort
#include <functional>   // std::greater
#include <vector>

namespace stan {

  namespace math {

    template <typename T>
    inline
    std::vector< fvar<T> >
    sort_desc(std::vector< fvar<T> > xs) {
      std::sort(xs.begin(), xs.end(), std::greater< fvar<T> >());
      return xs;
    }

    template <typename T, int R, int C>
    inline
    typename Eigen::Matrix<fvar<T>, R, C>
    sort_desc(Eigen::Matrix<fvar<T>, R, C> xs) {
      std::sort(xs.data(), xs.data()+xs.size(), std::greater< fvar<T> >());
      return xs;
    }

  }
}
#endif
