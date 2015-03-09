#ifndef STAN__MATH__FWD__MAT__FUN__SORT_ASC_HPP
#define STAN__MATH__FWD__MAT__FUN__SORT_ASC_HPP

#include <stan/math/fwd/core.hpp>

#include <vector>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <algorithm>    // std::sort
#include <functional>   // std::greater

namespace stan {

  namespace agrad {

    template <typename T>
    inline
    std::vector< fvar<T> >
    sort_asc(std::vector< fvar<T> > xs) {
      std::sort(xs.begin(), xs.end());      
      return xs;      
    }

    template <typename T, int R, int C>
    inline
    typename Eigen::Matrix<fvar<T>,R,C>
    sort_asc(Eigen::Matrix<fvar<T>,R,C> xs) {
      std::sort(xs.data(), xs.data()+xs.size());      
      return xs;      
    }

  }
}
#endif
