#ifndef __STAN__AGRAD__REV__MATRIX__ROWS_DOT_SELF_HPP__
#define __STAN__AGRAD__REV__MATRIX__ROWS_DOT_SELF_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_vector.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/dot_self.hpp>

namespace stan {
  namespace agrad {

    template<typename T, int R,int C>
    inline Eigen::Matrix<fvar<T>,R,1> 
    rows_dot_self(const Eigen::Matrix<fvar<T>,R,C>& x) {
      Eigen::Matrix<fvar<T>,R,1> ret(x.rows(),1);
      for (size_type i = 0; i < x.rows(); i++) {
        Eigen::Matrix<fvar<T>,1,C> crow = x.row(i);
        ret(i,0) = dot_self(crow);
      }
      return ret;
    }
  }
}
#endif
