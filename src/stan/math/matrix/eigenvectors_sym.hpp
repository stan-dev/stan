#ifndef __STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP__
#define __STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_nonzero_size.hpp>
#include <stan/math/matrix/validate_symmetric.hpp>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    eigenvectors_sym(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      validate_nonzero_size(m,"eigenvectors_sym");
      validate_symmetric(m,"eigenvectors_sym");
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
        solver(m);
      return solver.eigenvectors(); 
    }

  }
}
#endif
