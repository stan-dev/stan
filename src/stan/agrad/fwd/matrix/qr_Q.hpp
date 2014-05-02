#ifndef __STAN__AGRAD__FWD__MATRIX__QR_Q_HPP__
#define __STAN__AGRAD__FWD__MATRIX__QR_Q_HPP__

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
    qr_Q(const Eigen::Matrix<fvar<T>,Eigen::Dynamic,Eigen::Dynamic>& m) {
      typedef Eigen::Matrix<fvar<T>,Eigen::Dynamic,Eigen::Dynamic> matrix_fwd_t;
      stan::math::validate_nonzero_size(m,"qr_Q");
      stan::math::validate_greater_or_equal(m.rows(),m.cols(),"m.rows()", "m.cols()", "qr_Q");
      Eigen::HouseholderQR< matrix_fwd_t > qr(m.rows(), m.cols());
      qr.compute(m);
      matrix_fwd_t Q = qr.householderQ();
      for (int i=0; i<m.cols(); i++)
        if (qr.matrixQR()(i,i) < 0.0)
          Q.col(i) *= -1.0;
      return Q;
    }
  }
}
#endif
