#ifndef __STAN__DIFF__FWD__MATRIX__COLUMNS_DOT_SELF_HPP__
#define __STAN__DIFF__FWD__MATRIX__COLUMNS_DOT_SELF_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/matrix/dot_self.hpp>

namespace stan {
  namespace diff {

    template<typename T, int R,int C>
    inline Eigen::Matrix<fvar<T>,1,C> 
    columns_dot_self(const Eigen::Matrix<fvar<T>,R,C>& x) {
      Eigen::Matrix<fvar<T>,1,C> ret(1,x.cols());
      for (size_type i = 0; i < x.cols(); i++) {
        Eigen::Matrix<fvar<T>,R,1> ccol = x.col(i);
        ret(0,i) = dot_self(ccol);
      }
      return ret;
    }    
  }
}
#endif
