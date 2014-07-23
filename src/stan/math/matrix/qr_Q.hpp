#ifndef STAN__MATH__MATRIX__QR_Q_HPP
#define STAN__MATH__MATRIX__QR_Q_HPP

#include <stan/math/error_handling/check_greater_or_equal.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <Eigen/QR>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    qr_Q(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> matrix_t;
      stan::math::check_nonzero_size("qr_Q(%1%)",m,"m",(double*)0);
      stan::math::check_greater_or_equal("qr_Q(%1%)",static_cast<size_t>(m.rows()),
                                         static_cast<size_t>(m.cols()),"m.rows()",
                                         (double*)0);

      Eigen::HouseholderQR<matrix_t> qr(m.rows(), m.cols());
      qr.compute(m);
      matrix_t Q = qr.householderQ();
      for (int i = 0; i < m.cols(); i++) 
        if (qr.matrixQR()(i,i) < 0) 
          Q.col(i) *= -1.0;
      return Q;
    }

  }
}
#endif
