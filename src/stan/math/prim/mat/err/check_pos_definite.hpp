#ifndef STAN_MATH_PRIM_MAT_ERR_CHECK_POS_DEFINITE_HPP
#define STAN_MATH_PRIM_MAT_ERR_CHECK_POS_DEFINITE_HPP

#include <stan/math/prim/mat/meta/get.hpp>
#include <stan/math/prim/mat/meta/length.hpp>
#include <stan/math/prim/mat/meta/is_vector.hpp>
#include <stan/math/prim/mat/meta/is_vector_like.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/prim/mat/err/constraint_tolerance.hpp>
#include <stan/math/prim/scal/err/check_positive_size.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/value_of_rec.hpp>
namespace stan {

  namespace math {
    using Eigen::Dynamic;

    /**
     * Return <code>true</code> if the specified square, symmetric
     * matrix is positive definite.
     *
     * @tparam T_y Type of scalar of the matrix
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param y Matrix to test
     *
     * @return <code>true</code> if the matrix is positive definite
     * @throw <code>std::invalid_argument</code> if the matrix is not square
     *   or if the matrix has 0 size.
     * @throw <code>std::domain_error</code> if the matrix is not symmetric,
     *   if it is not positive definite, or if any element is <code>NaN</code>.
     */
    template <typename T_y>
    inline bool
    check_pos_definite(const char* function,
                       const char* name,
                       const Eigen::Matrix<T_y, Dynamic, Dynamic>& y) {
      check_symmetric(function, name, y);
      check_positive_size(function, name, "rows", y.rows());

      if (y.rows() == 1 && !(y(0, 0) > CONSTRAINT_TOLERANCE))
        domain_error(function, name, y, "is not positive definite: ");

      using Eigen::LDLT;
      using Eigen::Matrix;
      using Eigen::Dynamic;
      LDLT< Matrix<double, Dynamic, Dynamic> > cholesky
        = value_of_rec(y).ldlt();
      if (cholesky.info() != Eigen::Success
          || !cholesky.isPositive()
          || (cholesky.vectorD().array() <= 0.0).any())
        domain_error(function, name, y, "is not positive definite:\n");
      check_not_nan(function, name, y);
      return true;
    }

    /**
     * Return <code>true</code> if the specified LDLT transform of a matrix
     * is positive definite.
     *
     * @tparam Derived Derived type of the Eigen::LDLT transform.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param cholesky Eigen::LDLT to test, whose progenitor 
     * must not have any NaN elements
     *
     * @return <code>true</code> if the matrix is positive definite
     * @throw <code>std::domain_error</code> if the matrix is not positive definite.
     */
    template <typename Derived>
    inline bool
    check_pos_definite(const char* function,
                       const char* name,
                       const Eigen::LDLT<Derived>& cholesky) {
      if (cholesky.info() != Eigen::Success
          || !cholesky.isPositive()
          || !(cholesky.vectorD().array() > 0.0).all())
        domain_error(function, "LDLT decomposition of", " failed", name);
      return true;
    }
    
    /**
     * Return <code>true</code> if the specified LLT transform of a matrix
     * is positive definite.
     *
     * @tparam Derived Derived type of the Eigen::LLT transform.
     *
     * @param function Function name (for error messages)
     * @param name Variable name (for error messages)
     * @param cholesky Eigen::LLT to test, whose progenitor 
     * must not have any NaN elements
     *
     * @return <code>true</code> if the matrix is positive definite
     * @throw <code>std::domain_error</code> if the diagonal of the L matrix is not positive.
     */
    template <typename Derived>
    inline bool
    check_pos_definite(const char* function,
                       const char* name,
                       const Eigen::LLT<Derived>& cholesky) {
      if (cholesky.info() != Eigen::Success
          || !(cholesky.matrixLLT().diagonal().array() > 0.0).all())
        domain_error(function, "Cholesky decomposition of", " failed", name);
      return true;
    }
    
  }
}
#endif
