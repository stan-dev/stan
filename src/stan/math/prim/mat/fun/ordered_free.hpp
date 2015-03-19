#ifndef STAN__MATH__PRIM__MAT__FUN__ORDERED_FREE_HPP
#define STAN__MATH__PRIM__MAT__FUN__ORDERED_FREE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_ordered.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <cmath>

namespace stan {

  namespace prob {

    /**
     * Return the vector of unconstrained scalars that transform to
     * the specified positive ordered vector.
     *
     * <p>This function inverts the constraining operation defined in
     * <code>ordered_constrain(Matrix)</code>,
     *
     * @param y Vector of positive, ordered scalars.
     * @return Free vector that transforms into the input vector.
     * @tparam T Type of scalar.
     * @throw std::domain_error if y is not a vector of positive,
     *   ordered scalars.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    ordered_free(const Eigen::Matrix<T,Eigen::Dynamic,1>& y) {
      stan::math::check_ordered("stan::prob::ordered_free",
                                          "Ordered variable", y);
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = y.size();
      Matrix<T,Dynamic,1> x(k);
      if (k == 0)
        return x;
      x[0] = y[0];
      for (size_type i = 1; i < k; ++i)
        x[i] = log(y[i] - y[i-1]);
      return x;
    }

  }

}

#endif
