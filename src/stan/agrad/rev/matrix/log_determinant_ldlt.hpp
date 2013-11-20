#ifndef __STAN__AGRAD__REV__MATRIX__LOG_DETERMINANT_LDLT_HPP__
#define __STAN__AGRAD__REV__MATRIX__LOG_DETERMINANT_LDLT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>
#include <stan/agrad/rev/matrix/LDLT_alloc.hpp>
#include <stan/agrad/rev/matrix/LDLT_factor.hpp>

namespace stan {
  namespace agrad {
    namespace {

      template<int R,int C>
      class log_det_ldlt_vari : public vari {
      public:
        log_det_ldlt_vari(const stan::math::LDLT_factor<var,R,C> &A)
          : vari(A._alloc->log_abs_det()), _alloc_ldlt(A._alloc)
        { }

        virtual void chain() {
          Eigen::Matrix<double,R,C> invA;
          
          // If we start computing Jacobians, this may be a bit inefficient
          invA.setIdentity(_alloc_ldlt->N_, _alloc_ldlt->N_);
          _alloc_ldlt->_ldlt.solveInPlace(invA);

          for (size_t j = 0; j < _alloc_ldlt->N_; j++) {
            for (size_t i = 0; i < _alloc_ldlt->N_; i++) {
              _alloc_ldlt->_variA(i,j)->adj_ += adj_ * invA(i,j);
            }
          }
        }
        
        const LDLT_alloc<R,C> *_alloc_ldlt;
      };
    }

    template<int R, int C>
    var log_determinant_ldlt(stan::math::LDLT_factor<var,R,C> &A) {
      return var(new log_det_ldlt_vari<R,C>(A));
    }
  }
}
#endif
