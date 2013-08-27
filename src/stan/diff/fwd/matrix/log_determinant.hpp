#ifndef __STAN__DIFF__FWD__MATRIX__LOG_DETERMINANT_HPP__
#define __STAN__DIFF__FWD__MATRIX__LOG_DETERMINANT_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/diff/fwd/matrix/determinant.hpp>
#include <stan/diff/fwd/fabs.hpp>
#include <stan/diff/fwd/log.hpp>

namespace stan {
  namespace diff {
    
    template<typename T, int R,int C>
    inline 
    fvar<T>
    log_determinant(const Eigen::Matrix<fvar<T>, R, C>& m) {
      stan::math::validate_square(m, "log_determinant");

      return stan::diff::log(stan::diff::fabs(stan::diff::determinant(m)));
    }
  }
}
#endif
