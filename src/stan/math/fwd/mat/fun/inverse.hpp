#ifndef STAN__MATH__FWD__MAT__FUN__INVERSE_HPP
#define STAN__MATH__FWD__MAT__FUN__INVERSE_HPP

#include <vector>
#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/fwd/core.hpp>
#include <stan/math/fwd/mat/fun/to_fvar.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/inverse.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace agrad {

    template<typename T, int R,int C>
    inline
    Eigen::Matrix<fvar<T>,R,C>
    inverse(const Eigen::Matrix<fvar<T>, R, C>& m) {
      using stan::math::multiply;
      using stan::agrad::multiply;
      using stan::math::inverse;
      stan::math::check_square("inverse", "m", m);
      Eigen::Matrix<T,R,C> m_deriv(m.rows(), m.cols());
      Eigen::Matrix<T,R,C> m_inv(m.rows(), m.cols());

      for(size_type i = 0; i < m.rows(); i++) {
        for(size_type j = 0; j < m.cols(); j++) {
          m_inv(i,j) = m(i,j).val_;
          m_deriv(i,j) = m(i,j).d_;
        }
      }

      m_inv = stan::math::inverse(m_inv);

      m_deriv = multiply(multiply(m_inv, m_deriv), m_inv);
      m_deriv = -m_deriv;

      return to_fvar(m_inv, m_deriv);
    }
  }
}
#endif
