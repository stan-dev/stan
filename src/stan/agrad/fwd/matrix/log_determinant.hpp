#ifndef STAN__AGRAD__FWD__MATRIX__LOG_DETERMINANT_HPP
#define STAN__AGRAD__FWD__MATRIX__LOG_DETERMINANT_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/matrix/typedefs.hpp>
#include <stan/agrad/fwd/matrix/determinant.hpp>
#include <stan/agrad/fwd/functions/fabs.hpp>
#include <stan/agrad/fwd/functions/log.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace agrad {
    
    template<typename T, int R,int C>
    inline 
    fvar<T>
    log_determinant(const Eigen::Matrix<fvar<T>, R, C>& m) {
      stan::error_handling::check_square("log_determinant", "m", m);

      return stan::agrad::log(stan::agrad::fabs(stan::agrad::determinant(m)));
    }
  }
}
#endif
