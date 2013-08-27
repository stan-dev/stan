#ifndef __STAN__DIFF__FWD__MATRIX__DETERMINANT_HPP__
#define __STAN__DIFF__FWD__MATRIX__DETERMINANT_HPP__

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/diff/fwd/fvar.hpp>
#include <stan/diff/fwd/matrix/typedefs.hpp>
#include <stan/math/matrix/multiply.hpp>

namespace stan {
  namespace diff {
    
    template<typename T, int R,int C>
    inline 
    fvar<T>
    determinant(const Eigen::Matrix<fvar<T>, R, C>& m) {
      stan::math::validate_square(m, "determinant");
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

      m_inv = m_val.inverse();
      m_deriv = stan::math::multiply(m_inv, m_deriv);

      result.val_ = m_val.determinant();
      result.d_ = m_val.determinant() * m_deriv.trace();

      return result;
    }
  }
}
#endif
