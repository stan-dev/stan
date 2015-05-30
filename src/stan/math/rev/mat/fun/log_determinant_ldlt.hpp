#ifndef STAN_MATH_REV_MAT_FUN_LOG_DETERMINANT_LDLT_HPP
#define STAN_MATH_REV_MAT_FUN_LOG_DETERMINANT_LDLT_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/mat/fun/LDLT_alloc.hpp>
#include <stan/math/rev/mat/fun/LDLT_factor.hpp>

namespace stan {
  namespace math {
    namespace {

    /**
     * Returns the log det of the matrix whose LDLT factorization is given
     * See The Matrix Cookbook's chapter on Derivatives of a Determinant
     * In this case, it is just the inverse of the underlying matrix
     * @param A, which is a LDLT_factor
     * @return ln(det(A))
     * @throws never
     */

      template<int R, int C>
      class log_det_ldlt_vari : public vari {
      public:
        explicit log_det_ldlt_vari(const stan::math::LDLT_factor<var, R, C> &A)
          : vari(A._alloc->log_abs_det()), _alloc_ldlt(A._alloc)
        { }

        virtual void chain() {
          Eigen::Matrix<double, R, C> invA;

          // If we start computing Jacobians, this may be a bit inefficient
          invA.setIdentity(_alloc_ldlt->N_, _alloc_ldlt->N_);
          _alloc_ldlt->_ldlt.solveInPlace(invA);

          for (size_t j = 0; j < _alloc_ldlt->N_; j++) {
            for (size_t i = 0; i < _alloc_ldlt->N_; i++) {
              _alloc_ldlt->_variA(i, j)->adj_ += adj_ * invA(i, j);
            }
          }
        }

        const LDLT_alloc<R, C> *_alloc_ldlt;
      };
    }

    template<int R, int C>
    var log_determinant_ldlt(stan::math::LDLT_factor<var, R, C> &A) {
      return var(new log_det_ldlt_vari<R, C>(A));
    }
  }
}
#endif
