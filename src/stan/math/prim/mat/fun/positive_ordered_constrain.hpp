#ifndef STAN__MATH__PRIM__MAT__FUN__POSITIVE_ORDERED_CONSTRAIN_HPP
#define STAN__MATH__PRIM__MAT__FUN__POSITIVE_ORDERED_CONSTRAIN_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <cmath>

namespace stan {

  namespace prob {

    /**
     * Return an increasing positive ordered vector derived from the specified
     * free vector.  The returned constrained vector will have the
     * same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @return Positive, increasing ordered vector.
     * @tparam T Type of scalar.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    positive_ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = x.size();
      Matrix<T,Dynamic,1> y(k);
      if (k == 0)
        return y;
      y[0] = exp(x[0]);
      for (size_type i = 1; i < k; ++i)
        y[i] = y[i-1] + exp(x[i]);
      return y;
    }

    /**
     * Return a positive valued, increasing positive ordered vector derived
     * from the specified free vector and increment the specified log
     * probability reference with the log absolute Jacobian determinant
     * of the transform.  The returned constrained vector
     * will have the same dimensionality as the specified free vector.
     *
     * @param x Free vector of scalars.
     * @param lp Log probability reference.
     * @return Positive, increasing ordered vector.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline
    Eigen::Matrix<T,Eigen::Dynamic,1>
    positive_ordered_constrain(const Eigen::Matrix<T,Eigen::Dynamic,1>& x, T& lp) {
      using Eigen::Matrix;
      using Eigen::Dynamic;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      for (size_type i = 0; i < x.size(); ++i)
        lp += x(i);
      return positive_ordered_constrain(x);
    }

  }

}

#endif
