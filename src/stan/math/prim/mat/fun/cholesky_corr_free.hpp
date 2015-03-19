#ifndef STAN__MATH__PRIM__MAT__FUN__CHOLESKY_CORR_FREE_HPP
#define STAN__MATH__PRIM__MAT__FUN__CHOLESKY_CORR_FREE_HPP

#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/scal/fun/corr_constrain.hpp>
#include <stan/math/prim/scal/fun/square.hpp>
#include <stan/math/prim/scal/fun/corr_free.hpp>
#include <cmath>

namespace stan {

  namespace prob {


    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    cholesky_corr_free(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& x) {
      using std::sqrt;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::square;

      stan::math::check_square("cholesky_corr_free", "x", x);
      // should validate lower-triangular, unit lengths

      int K = (x.rows() * (x.rows() - 1)) / 2;
      Matrix<T,Dynamic,1> z(K);
      int k = 0;
      for (int i = 1; i < x.rows(); ++i) {
        z(k++) = corr_free(x(i,0));
        double sum_sqs = square(x(i,0));
        for (int j = 1; j < i; ++j) {
          z(k++) = corr_free(x(i,j) / sqrt(1.0 - sum_sqs));
          sum_sqs += square(x(i,j));
        }
      }
      return z;
    }
  }

}

#endif
