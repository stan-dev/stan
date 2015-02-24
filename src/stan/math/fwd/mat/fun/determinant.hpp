#ifndef STAN__MATH__FWD__MAT__FUN__DETERMINANT_HPP
#define STAN__MATH__FWD__MAT__FUN__DETERMINANT_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/typedefs.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/inverse.hpp>
#include <stan/math/fwd/mat/fun/inverse.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace agrad {
    
    template<typename T, int R,int C>
    inline 
    fvar<T>
    determinant(const Eigen::Matrix<fvar<T>, R, C>& m) {
      using stan::math::multiply;

      stan::math::check_square("determinant", "m", m);
      Eigen::Matrix<T,R,C> m_deriv(m.rows(), m.cols());
      Eigen::Matrix<T,R,C> m_val(m.rows(), m.cols());
      Eigen::Matrix<T,R,C> m_inv(m.rows(), m.cols());
      fvar<T> result;

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; j < m.cols(); j++) {
          m_deriv(i,j) = m(i,j).d_;
          m_val(i,j) = m(i,j).val_;
        }
      }

      m_inv = stan::math::inverse(m_val);
      m_deriv = multiply(m_inv, m_deriv);

      result.val_ = m_val.determinant();
      result.d_ = result.val_ * m_deriv.trace();

      return result;
    }
  }
}
#endif
