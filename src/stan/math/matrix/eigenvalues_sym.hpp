#ifndef __STAN__MATH__MATRIX__EIGENVALUES_SYM_HPP__
#define __STAN__MATH__MATRIX__EIGENVALUES_SYM_HPP__

#include <stan/math/error_handling/matrix/check_nonzero_size.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/fwd/functions/abs.hpp>
#include <stan/agrad/rev/functions/abs.hpp>
#include <stan/agrad/fwd/functions/fabs.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/fwd/functions/sqrt.hpp>
#include <stan/agrad/rev/functions/sqrt.hpp>

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
      stan::math::check_nonzero_size("eigenvalues_sym(%1%)",m,
                                     "m",(double*)0);
      stan::math::check_symmetric("eigenvalues_sym(%1%)",m,"m",(double*)0);

      Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> >
        solver(m,Eigen::EigenvaluesOnly);
      return solver.eigenvalues();
    }

  }
}
#endif
