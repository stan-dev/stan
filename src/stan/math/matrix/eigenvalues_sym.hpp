#ifndef __STAN__MATH__MATRIX__EIGENVALUES_SYM_HPP__
#define __STAN__MATH__MATRIX__EIGENVALUES_SYM_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_nonzero_size.hpp>
#include <stan/math/matrix/validate_symmetric.hpp>

namespace stan {
  namespace math {

    /**
     * Return the eigenvalues of the specified symmetric matrix
     * in descending order of magnitude.  This function is more
     * efficient than the general eigenvalues function for symmetric
     * matrices.
     * <p>See <code>eigen_decompose()</code> for more information.
     * @param m Specified matrix.
     * @return Eigenvalues of matrix.
     */
    template <typename T>
    Eigen::Matrix<T,Eigen::Dynamic,1>
    eigenvalues_sym(const Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>& m) {
      validate_nonzero_size(m,"eigenvalues_sym");
      validate_symmetric(m,"eigenvalues_sym");
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
        solver(m,Eigen::EigenvaluesOnly);
      return solver.eigenvalues();
    }

  }
}
#endif
