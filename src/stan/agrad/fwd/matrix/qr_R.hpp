#ifndef __STAN__AGRAD__FWD__MATRIX__QR_R_HPP__
#define __STAN__AGRAD__FWD__MATRIX__QR_R_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <Eigen/QR>
#include <stan/math/matrix/validate_nonzero_size.hpp>
#include <stan/math/matrix/validate_greater_or_equal.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/operators/operator_less_than.hpp>

namespace stan {
  namespace agrad {

    template <typename T>
    Eigen::Matrix<fvar<T>,Eigen::Dynamic,Eigen::Dynamic>
    qr_R(const Eigen::Matrix<fvar<T>,Eigen::Dynamic,Eigen::Dynamic>& m) {
      typedef Eigen::Matrix<fvar<T>,Eigen::Dynamic,Eigen::Dynamic> matrix_fwd_t;
      stan::math::validate_nonzero_size(m,"qr_R");
      stan::math::validate_greater_or_equal(m.rows(),m.cols(),"m.rows()", "m.cols()", "qr_R");
      Eigen::HouseholderQR< matrix_fwd_t > qr(m.rows(), m.cols());
      qr.compute(m);
      matrix_fwd_t R = qr.matrixQR().topLeftCorner(m.rows(),m.cols());
      for (int i=0; i<R.rows(); i++) {
        for (int j=0; j<i; j++)
          R(i,j) = 0.0;
        if (i < R.cols() && R(i,i) < 0.0)
          R.row(i) *= -1.0;
      }
      return R;
    }
  }
}
#endif
