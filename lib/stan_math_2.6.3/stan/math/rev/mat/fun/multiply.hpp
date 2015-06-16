#ifndef STAN_MATH_REV_MAT_FUN_MULTIPLY_HPP
#define STAN_MATH_REV_MAT_FUN_MULTIPLY_HPP

#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/Eigen_NumTraits.hpp>
#include <stan/math/rev/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/rev/mat/fun/to_var.hpp>
#include <stan/math/rev/mat/fun/dot_product.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {

    /**
     * Return the product of two scalars.
     * @param[in] v First scalar.
     * @param[in] c Specified scalar.
     * @return Product of scalars.
     */
    template <typename T1, typename T2>
    inline typename
    boost::enable_if_c<
      (boost::is_scalar<T1>::value || boost::is_same<T1, var>::value)
      && (boost::is_scalar<T2>::value || boost::is_same<T2, var>::value),
      typename boost::math::tools::promote_args<T1, T2>::type>::type
    multiply(const T1& v, const T2& c) {
      return v * c;
    }

    /**
     * Return the product of scalar and matrix.
     * @param[in] c Specified scalar.
     * @param[in] m Matrix.
     * @return Product of scalar and matrix.
     */
    template<typename T1, typename T2, int R2, int C2>
    inline Eigen::Matrix<var, R2, C2>
    multiply(const T1& c, const Eigen::Matrix<T2, R2, C2>& m) {
      // FIXME:  pull out to eliminate overpromotion of one side
      // move to matrix.hpp w. promotion?
      return to_var(m) * to_var(c);
    }

    /**
     * Return the product of scalar and matrix.
     * @param[in] m Matrix.
     * @param[in] c Specified scalar.
     * @return Product of scalar and matrix.
     */
    template<typename T1, int R1, int C1, typename T2>
    inline Eigen::Matrix<var, R1, C1>
    multiply(const Eigen::Matrix<T1, R1, C1>& m, const T2& c) {
      return to_var(m) * to_var(c);
    }

    /**
     * Return the product of the specified matrices.  The number of
     * columns in the first matrix must be the same as the number of rows
     * in the second matrix.
     * @param[in] m1 First matrix.
     * @param[in] m2 Second matrix.
     * @return The product of the first and second matrices.
     * @throw std::domain_error if the number of columns of m1 does not match
     *   the number of rows of m2.
     */
    template<typename T1, int R1, int C1, typename T2, int R2, int C2>
    inline typename
    boost::enable_if_c< boost::is_same<T1, var>::value ||
                        boost::is_same<T2, var>::value,
                        Eigen::Matrix<var, R1, C2> >::type
    multiply(const Eigen::Matrix<T1, R1, C1>& m1,
             const Eigen::Matrix<T2, R2, C2>& m2) {
      stan::math::check_multiplicable("multiply",
                                                "m1", m1,
                                                "m2", m2);
      Eigen::Matrix<var, R1, C2> result(m1.rows(), m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<T1, R1, C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<T2, R2, C2>::ConstColXpr ccol(m2.col(j));
          if (j == 0) {
            if (i == 0) {
              result(i, j) = var(new dot_product_vari<T1, T2>(crow, ccol));
            } else {
              dot_product_vari<T1, T2> *v2
                = static_cast<dot_product_vari<T1, T2>*>(result(0, j).vi_);
              result(i, j)
                = var(new dot_product_vari<T1, T2>(crow, ccol, NULL, v2));
            }
          } else {
            if (i == 0) {
              dot_product_vari<T1, T2> *v1
                = static_cast<dot_product_vari<T1, T2>*>(result(i, 0).vi_);
              result(i, j)
                = var(new dot_product_vari<T1, T2>(crow, ccol, v1, NULL));
            } else /* if (i != 0 && j != 0) */ {
              dot_product_vari<T1, T2> *v1
                = static_cast<dot_product_vari<T1, T2>*>(result(i, 0).vi_);
              dot_product_vari<T1, T2> *v2
                = static_cast<dot_product_vari<T1, T2>*>(result(0, j).vi_);
              result(i, j)
                = var(new dot_product_vari<T1, T2>(crow, ccol, v1, v2));
            }
          }
        }
      }
      return result;
    }

    /**
     * Return the scalar product of the specified row vector and
     * specified column vector.  The return is the same as the dot
     * product.  The two vectors must be the same size.
     * @param[in] rv Row vector.
     * @param[in] v Column vector.
     * @return Scalar result of multiplying row vector by column vector.
     * @throw std::domain_error if rv and v are not the same size
     */
    template <typename T1, int C1, typename T2, int R2>
    inline typename
    boost::enable_if_c< boost::is_same<T1, var>::value ||
                        boost::is_same<T2, var>::value, var >::type
    multiply(const Eigen::Matrix<T1, 1, C1>& rv,
             const Eigen::Matrix<T2, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::domain_error("row vector and vector must be same length "
                                "in multiply");
      return dot_product(rv, v);
    }

  }
}
#endif
