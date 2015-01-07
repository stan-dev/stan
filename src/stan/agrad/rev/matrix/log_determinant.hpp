#ifndef STAN__AGRAD__REV__MATRIX__LOG_DETERMINANT_HPP
#define STAN__AGRAD__REV__MATRIX__LOG_DETERMINANT_HPP

#include <stan/agrad/rev.hpp> 
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_square.hpp>

namespace stan {

  namespace agrad {

    template <int R, int C>
    inline var log_determinant(const Eigen::Matrix<var,R,C>& m) {
      using Eigen::Matrix;

      error_handling::check_square("log_determinant","m", m);

      Matrix<double,R,C> m_d(m.rows(),m.cols());
      for (int i = 0; i < m.size(); ++i)
        m_d(i) = m(i).val();

      Eigen::FullPivHouseholderQR<Matrix<double,R,C> > hh
        = m_d.fullPivHouseholderQr();

      double val = hh.logAbsDeterminant();

      vari** varis 
        = ChainableStack::memalloc_.alloc_array<vari*>(m.size());
      for (int i = 0; i < m.size(); ++i)
        varis[i] = m(i).vi_;

      Matrix<double,R,C> m_inv_transpose = hh.inverse().transpose();
      double* gradients 
        = ChainableStack::memalloc_.alloc_array<double>(m.size());
      for (int i = 0; i < m.size(); ++i)
        gradients[i] = m_inv_transpose(i);

      return var(new precomputed_gradients_vari(val,m.size(),varis,gradients));
    }
    
  }
}
#endif
