#ifndef STAN_MATH_PRIM_MAT_FUN_UNIT_VECTOR_FREE_HPP
#define STAN_MATH_PRIM_MAT_FUN_UNIT_VECTOR_FREE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/mat/err/check_unit_vector.hpp>
#include <cmath>

namespace stan {

  namespace math {


    template <typename T>
    Eigen::Matrix<T, Eigen::Dynamic, 1>
    unit_vector_free(const Eigen::Matrix<T, Eigen::Dynamic, 1>& x) {
      using Eigen::Matrix;
      using Eigen::Dynamic;

      stan::math::check_unit_vector("stan::math::unit_vector_free",
                                              "Unit vector variable", x);
      int Km1 = x.size() - 1;
      Matrix<T, Dynamic, 1> y(Km1);
      T sumSq = x(Km1)*x(Km1);
      const T half_pi = T(M_PI/2.0);
      for (int k = Km1; --k >= 0; ) {
        y(k) = atan2(sqrt(sumSq), x(k)) - half_pi;
        sumSq += x(k)*x(k);
      }
      return y;
    }

  }

}

#endif
