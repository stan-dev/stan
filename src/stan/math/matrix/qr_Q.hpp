#ifndef __STAN__MATH__MATRIX__QR_Q_HPP__
#define __STAN__MATH__MATRIX__QR_Q_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <Eigen/QR>
#include <stan/math/matrix/validate_nonzero_size.hpp>
#include <stan/math/matrix/validate_greater_or_equal.hpp>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    qr_Q(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      validate_nonzero_size(m,"qr_Q");
      validate_greater_or_equal(m.rows(),m.cols(),"m.rows()", "m.cols()", "qr_Q");
      Eigen::HouseholderQR< Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> > qr(m.rows(), m.cols());
      qr.compute(m);
      Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> Q = qr.householderQ();
      const T zero = 0.0;
      for (int i=0; i<m.cols(); i++) if (qr.matrixQR()(i,i) < zero) Q.col(i) *= -1.0;
      return Q;
    }
  }
}
#endif
