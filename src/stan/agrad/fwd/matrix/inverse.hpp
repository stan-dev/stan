#ifndef STAN__AGRAD__FWD__MATRIX__INVERSE_HPP
#define STAN__AGRAD__FWD__MATRIX__INVERSE_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/agrad/fwd/fvar.hpp>
#include <stan/agrad/fwd/operators/operator_multiplication.hpp>
#include <stan/agrad/fwd/matrix/to_fvar.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/inverse.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace agrad {
    
    template<typename T, int R,int C>
    inline 
    Eigen::Matrix<fvar<T>,R,C> 
    inverse(const Eigen::Matrix<fvar<T>, R, C>& m) {
      using stan::math::multiply;
      using stan::agrad::multiply;
      using stan::math::inverse;
      stan::error_handling::check_square("inverse", "m", m);
      Eigen::Matrix<T,R,C> m_deriv(m.rows(), m.cols());
      Eigen::Matrix<T,R,C> m_inv(m.rows(), m.cols());

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; j < m.cols(); j++) {
          m_inv(i,j) = m(i,j).val_;
          m_deriv(i,j) = m(i,j).d_;
        }
      }

      m_inv = stan::math::inverse(m_inv);

      m_deriv = -1 * multiply(multiply(m_inv, m_deriv), m_inv);
    
      return to_fvar(m_inv, m_deriv);
    }
  }
}
#endif
