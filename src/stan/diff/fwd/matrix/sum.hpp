#ifndef __STAN__DIFF__FWD__MATRIX__SUM_HPP__
#define __STAN__DIFF__FWD__MATRIX__SUM_HPP__

#include <vector>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/diff/fwd/fvar.hpp>

namespace stan {
  namespace diff {

    template <typename T, int R, int C>
    inline 
    fvar<T> 
    sum(const Eigen::Matrix<fvar<T>,R,C>& m) {
      fvar<T> sum = 0;
      if (m.size() == 0)
        return 0.0;
      for(unsigned i = 0; i < m.rows(); i++) {
        for(unsigned j = 0; j < m.cols(); j++)
          sum += m(i,j);
      }
      return sum;
    }
  }
}
#endif
