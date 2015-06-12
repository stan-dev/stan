#ifndef STAN_MATH_FWD_MAT_FUN_QR_Q_HPP
#define STAN_MATH_FWD_MAT_FUN_QR_Q_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <Eigen/QR>
#include <stan/math/prim/scal/err/check_nonzero_size.hpp>
#include <stan/math/prim/scal/err/check_greater_or_equal.hpp>
#include <stan/math/fwd/core.hpp>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<fvar<T>, Eigen::Dynamic, Eigen::Dynamic>
    qr_Q(const Eigen::Matrix<fvar<T>, Eigen::Dynamic, Eigen::Dynamic>& m) {
      typedef Eigen::Matrix<fvar<T>, Eigen::Dynamic, Eigen::Dynamic>
        matrix_fwd_t;
      stan::math::check_nonzero_size("qr_Q", "m", m);
      stan::math::check_greater_or_equal("qr_Q", "m.rows()", m.rows(),
                                         m.cols());
      Eigen::HouseholderQR< matrix_fwd_t > qr(m.rows(), m.cols());
      qr.compute(m);
      matrix_fwd_t Q = qr.householderQ();
      for (int i = 0; i < m.cols(); i++)
        if (qr.matrixQR()(i, i) < 0.0)
          Q.col(i) *= -1.0;
      return Q;
    }
  }
}
#endif
