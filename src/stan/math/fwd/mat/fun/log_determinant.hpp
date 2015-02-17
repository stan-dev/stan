#ifndef STAN__MATH__FWD__MAT__FUN__LOG_DETERMINANT_HPP
#define STAN__MATH__FWD__MAT__FUN__LOG_DETERMINANT_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core/fvar.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/fwd/mat/fun/determinant.hpp>
#include <stan/math/fwd/scal/fun/fabs.hpp>
#include <stan/math/fwd/scal/fun/log.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace agrad {
    
    template<typename T, int R,int C>
    inline 
    fvar<T>
    log_determinant(const Eigen::Matrix<fvar<T>, R, C>& m) {
      stan::math::check_square("log_determinant", "m", m);

      return stan::agrad::log(stan::agrad::fabs(stan::agrad::determinant(m)));
    }
  }
}
#endif
