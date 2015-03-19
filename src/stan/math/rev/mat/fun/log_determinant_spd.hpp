#ifndef STAN__MATH__REV__MAT__FUN__LOG_DETERMINANT_SPD_HPP
#define STAN__MATH__REV__MAT__FUN__LOG_DETERMINANT_SPD_HPP

#include <boost/math/special_functions/fpclassify.hpp>
#include <stan/math/prim/scal/err/domain_error.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>

namespace stan {

  namespace agrad {

    template <int R, int C>
    inline var log_determinant_spd(const Eigen::Matrix<var,R,C>& m) {
      using stan::math::domain_error;
      using Eigen::Matrix;

      math::check_square("log_determinant_spd", "m", m);

      Matrix<double,R,C> m_d(m.rows(),m.cols());
      for (int i = 0; i < m.size(); ++i)
        m_d(i) = m(i).val();

      Eigen::LDLT<Matrix<double,R,C> > ldlt(m_d);
      if (ldlt.info() != Eigen::Success) {
        double y = 0;
        domain_error("log_determinant_spd",
                "matrix argument", y,
                "failed LDLT factorization");
      }

       // compute the inverse of A (needed for the derivative)
      m_d.setIdentity(m.rows(), m.cols());
      ldlt.solveInPlace(m_d);

      if (ldlt.isNegative() || (ldlt.vectorD().array() <= 1e-16).any()) {
        double y = 0;
        domain_error("log_determinant_spd",
                "matrix argument", y,
                "matrix is negative definite");
      }

      double val = ldlt.vectorD().array().log().sum();

      if (!boost::math::isfinite(val)) {
        double y = 0;
        domain_error("log_determinant_spd",
                "matrix argument", y,
                "log determininant is infinite");
      }

      vari** operands = ChainableStack::memalloc_
        .alloc_array<vari*>(m.size());
      for (int i = 0; i < m.size(); ++i)
        operands[i] = m(i).vi_;

      double* gradients = ChainableStack::memalloc_
        .alloc_array<double>(m.size());
      for (int i = 0; i < m.size(); ++i)
        gradients[i] = m_d(i);

      return var(new precomputed_gradients_vari(val,m.size(),operands,gradients));
    }


  }

}
#endif
