#ifndef STAN__MATH__PRIM__MAT__FUN__CORR_MATRIX_FREE_HPP
#define STAN__MATH__PRIM__MAT__FUN__CORR_MATRIX_FREE_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <boost/throw_exception.hpp>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace stan {

  namespace prob {

    const double CONSTRAINT_TOLERANCE = 1E-8;

    /**
     * Return the vector of unconstrained partial correlations that
     * define the specified correlation matrix when transformed.
     *
     * <p>The constraining transform is defined as for
     * <code>corr_matrix_constrain(Matrix,size_t)</code>.  The
     * inverse transform in this function is simpler in that it only
     * needs to compute the \f$k \choose 2\f$ partial correlations
     * and then free those.
     *
     * @param y The correlation matrix to free.
     * @return Vector of unconstrained values that produce the
     * specified correlation matrix when transformed.
     * @tparam T Type of scalar.
     * @throw std::domain_error if the correlation matrix has no
     *    elements or is not a square matrix.
     * @throw std::runtime_error if the correlation matrix cannot be
     *    factorized by factor_cov_matrix() or if the sds returned by
     *    factor_cov_matrix() on log scale are unconstrained.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    corr_matrix_free(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& y) {
      using Eigen::Array;
      using Eigen::Dynamic;
      using Eigen::Matrix;
      using stan::math::index_type;
      typedef typename index_type<Matrix<T,Dynamic,1> >::type size_type;

      size_type k = y.rows();
      if (y.cols() != k)
        throw std::domain_error("y is not a square matrix");
      if (k == 0)
        throw std::domain_error("y has no elements");
      size_type k_choose_2 = (k * (k-1)) / 2;
      Array<T,Dynamic,1> x(k_choose_2);
      Array<T,Dynamic,1> sds(k);
      bool successful = factor_cov_matrix(y,x,sds);
      if (!successful)
        throw std::runtime_error("factor_cov_matrix failed on y");
      for (size_type i = 0; i < k; ++i) {
        // sds on log scale unconstrained
        if (fabs(sds[i] - 0.0) >= CONSTRAINT_TOLERANCE) {
          std::stringstream s;
          s << "all standard deviations must be zero."
            << " found log(sd[" << i << "])=" << sds[i] << std::endl;
          BOOST_THROW_EXCEPTION(std::runtime_error(s.str()));
        }
      }
      return x.matrix();
    }
  }

}

#endif
