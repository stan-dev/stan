#ifndef STAN__MATH__MATRIX__QR_R_HPP
#define STAN__MATH__MATRIX__QR_R_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <Eigen/QR>
#include <stan/math/error_handling/check_greater_or_equal.hpp>
#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    qr_R(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      typedef Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> matrix_t;
      stan::math::check_nonzero_size("qr_R(%1%)",m,"m",(double*)0);
      stan::math::check_greater_or_equal("qr_R(%1%)",static_cast<size_t>(m.rows()),
                                         static_cast<size_t>(m.cols()),"m.rows()",
                                         (double*)0);
      Eigen::HouseholderQR<matrix_t> qr(m.rows(), m.cols());
      qr.compute(m);
      matrix_t R = qr.matrixQR().topLeftCorner(m.rows(),m.cols());
      for (int i=0; i<R.rows(); i++) {
        for (int j=0; j<i; j++)
          R(i,j) = 0.0;
        if (i < R.cols() && R(i,i) < 0)
          R.row(i) *= -1.0;
      }
      return R;
    }
  }
}
#endif
