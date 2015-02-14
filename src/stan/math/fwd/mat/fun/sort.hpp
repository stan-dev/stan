#ifndef STAN__MATH__FWD__MAT__FUN__SORT_HPP
#define STAN__MATH__FWD__MAT__FUN__SORT_HPP

#include <stan/math/fwd/scal/meta/fvar.hpp>
#include <stan/math/prim/scal/meta/traits.hpp>
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

    template <typename T>
    inline
    std::vector< fvar<T> >
    sort_desc(std::vector< fvar<T> > xs) {
      std::sort(xs.begin(), xs.end(), std::greater< fvar<T> >());      
      return xs;      
    }
    
    template <typename T, int R, int C>
    inline
    typename Eigen::Matrix<fvar<T>,R,C>
    sort_asc(Eigen::Matrix<fvar<T>,R,C> xs) {
      std::sort(xs.data(), xs.data()+xs.size());      
      return xs;      
    }

    template <typename T, int R, int C>
    inline
    typename Eigen::Matrix<fvar<T>,R,C>
    sort_desc(Eigen::Matrix<fvar<T>,R,C> xs) {
      std::sort(xs.data(), xs.data()+xs.size(), std::greater< fvar<T> >());      
      return xs;      
    }
        
  }
}
#endif
