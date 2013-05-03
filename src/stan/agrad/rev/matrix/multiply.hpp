#ifndef __STAN__AGRAD__REV__MATRIX__MULTIPLY_HPP__
#define __STAN__AGRAD__REV__MATRIX__MULTIPLY_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/typedefs.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_matching_sizes.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/typedefs.hpp>
#include <stan/agrad/rev/matrix/to_var.hpp>
#include <stan/agrad/rev/matrix/dot_product.hpp>
#include <stan/agrad/rev/operator_multiplication.hpp>

namespace stan {
  namespace agrad {
    
    /**
     * Return the product of two scalars.
     * @param[in] v First scalar.
     * @param[in] c Specified scalar.
     * @return Product of scalars.
     */
    template <typename T1, typename T2>
    inline
    typename boost::math::tools::promote_args<T1,T2>::type
    multiply(const T1& v, const T2& c) {
      return v * c;
    }

    /**
     * Return the product of scalar and matrix.
     * @param[in] c Specified scalar.
     * @param[in] m Matrix.
     * @return Product of scalar and matrix.
     */
    template<typename T1,typename T2,int R2,int C2>
    inline Eigen::Matrix<var,R2,C2> multiply(const T1& c, 
                                             const Eigen::Matrix<T2, R2, C2>& m) {
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
    template<typename T1,int R1,int C1,typename T2>
    inline Eigen::Matrix<var,R1,C1> multiply(const Eigen::Matrix<T1, R1, C1>& m, 
                                             const T2& c) {
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
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<var,R1,C1>& m1,
                                             const Eigen::Matrix<var,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<var,R1,C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<var,R2,C2>::ConstColXpr ccol(m2.col(j));
          if (j == 0) {
            if (i == 0) {
              result(i,j) = var(new dot_product_vari<var,var>(crow,ccol));
            }
            else {
              dot_product_vari<var,var> *v2 = static_cast<dot_product_vari<var,var>*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vari<var,var>(crow,ccol,NULL,v2));
            }
          }
          else { 
            if (i == 0) {
              dot_product_vari<var,var> *v1 = static_cast<dot_product_vari<var,var>*>(result(i,0).vi_);
              result(i,j) = var(new dot_product_vari<var,var>(crow,ccol,v1));
            }
            else /* if (i != 0 && j != 0) */ {
              dot_product_vari<var,var> *v1 = static_cast<dot_product_vari<var,var>*>(result(i,0).vi_);
              dot_product_vari<var,var> *v2 = static_cast<dot_product_vari<var,var>*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vari<var,var>(crow,ccol,v1,v2));
            }
          }
        }
      }
      return result;
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
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<double,R1,C1>& m1,
                                             const Eigen::Matrix<var,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<double,R1,C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<var,R2,C2>::ConstColXpr ccol(m2.col(j));
          //          result(i,j) = dot_product(crow,ccol);
          if (j == 0) {
            if (i == 0) {
              result(i,j) = var(new dot_product_vari<var,double>(ccol,crow));
            }
            else {
              dot_product_vari<var,double> *v2 = static_cast<dot_product_vari<var,double>*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vari<var,double>(ccol,crow,v2,NULL));
            }
          }
          else { 
            if (i == 0) {
              dot_product_vari<var,double> *v1 = static_cast<dot_product_vari<var,double>*>(result(i,0).vi_);
              result(i,j) = var(new dot_product_vari<var,double>(ccol,crow,NULL,v1));
            }
            else /* if (i != 0 && j != 0) */ {
              dot_product_vari<var,double> *v1 = static_cast<dot_product_vari<var,double>*>(result(i,0).vi_);
              dot_product_vari<var,double> *v2 = static_cast<dot_product_vari<var,double>*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vari<var,double>(ccol,crow,v2,v1));
            }
          }
        }
      }
      return result;
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
    template<int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<var,R1,C2> multiply(const Eigen::Matrix<var,R1,C1>& m1,
                                             const Eigen::Matrix<double,R2,C2>& m2) {
      stan::math::validate_multiplicable(m1,m2,"multiply");
      Eigen::Matrix<var,R1,C2> result(m1.rows(),m2.cols());
      for (int i = 0; i < m1.rows(); i++) {
        typename Eigen::Matrix<var,R1,C1>::ConstRowXpr crow(m1.row(i));
        for (int j = 0; j < m2.cols(); j++) {
          typename Eigen::Matrix<double,R2,C2>::ConstColXpr ccol(m2.col(j));
          //          result(i,j) = dot_product(crow,ccol);
          if (j == 0) {
            if (i == 0) {
              result(i,j) = var(new dot_product_vari<var,double>(crow,ccol));
            }
            else {
              dot_product_vari<var,double> *v2 = static_cast<dot_product_vari<var,double>*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vari<var,double>(crow,ccol,NULL,v2));
            }
          }
          else { 
            if (i == 0) {
              dot_product_vari<var,double> *v1 = static_cast<dot_product_vari<var,double>*>(result(i,0).vi_);
              result(i,j) = var(new dot_product_vari<var,double>(crow,ccol,v1,NULL));
            }
            else /* if (i != 0 && j != 0) */ {
              dot_product_vari<var,double> *v1 = static_cast<dot_product_vari<var,double>*>(result(i,0).vi_);
              dot_product_vari<var,double> *v2 = static_cast<dot_product_vari<var,double>*>(result(0,j).vi_);
              result(i,j) = var(new dot_product_vari<var,double>(crow,ccol,v1,v2));
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
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<var, 1, C1>& rv, 
                        const Eigen::Matrix<var, R2, 1>& v) {
      if (rv.size() != v.size())
        throw std::domain_error("row vector and vector must be same length in multiply");
      return dot_product(rv, v);
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
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<double, 1, C1>& rv, 
                        const Eigen::Matrix<var, R2, 1>& v) {
      stan::math::validate_multiplicable(rv,v,"multiply");
      return dot_product(rv, v);
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
    template <int C1,int R2>
    inline var multiply(const Eigen::Matrix<var, 1, C1>& rv, 
                        const Eigen::Matrix<double, R2, 1>& v) {
      stan::math::validate_multiplicable(rv,v,"multiply");
      return dot_product(rv, v);
    }

  }
}
#endif
