#ifndef STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP
#define STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP

#include <stan/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    eigenvectors_sym(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      stan::error_handling::check_nonzero_size("eigenvectors_sym", "m", m);
      stan::error_handling::check_symmetric("eigenvalues_sym", "m", m);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
        solver(m);
      return solver.eigenvectors(); 
    }

  }
}
#endif
