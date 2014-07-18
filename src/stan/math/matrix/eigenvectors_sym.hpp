#ifndef STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP
#define STAN__MATH__MATRIX__EIGENVECTORS_SYM_HPP

#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/matrix/Eigen.hpp>

namespace stan {
  namespace math {

    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>
    eigenvectors_sym(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      stan::math::check_nonzero_size("eigenvectors_sym(%1%)",m,
                                     "m",(double*)0);
      stan::math::check_symmetric("eigenvalues_sym(%1%)",m,"m",(double*)0);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
        solver(m);
      return solver.eigenvectors(); 
    }

  }
}
#endif
