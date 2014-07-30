#ifndef STAN__AGRAD__REV__MATRIX__LDLT_ALLOC_HPP
#define STAN__AGRAD__REV__MATRIX__LDLT_ALLOC_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {
    /**
     * This object stores the actual (double typed) LDLT factorization of
     * an Eigen::Matrix<var> along with pointers to its vari's which allow the
     * *_ldlt functions to save memory.  It is derived from a chainable_alloc
     * object so that it is allocated on the stack but does not have a chain()
     * function called.
     *
     * This class should only be instantiated as part of an LDLT_factor object
     * and is only used in *_ldlt functions.
     **/
    template<int R, int C>
    class LDLT_alloc : public chainable_alloc {
    public:
      LDLT_alloc() : N_(0) {}
      LDLT_alloc(const Eigen::Matrix<var,R,C> &A) : N_(0) {
        compute(A);
      }
      
      /**
       * Compute the LDLT factorization and store pointers to the 
       * vari's of the matrix entries to be used when chain() is
       * called elsewhere.
       **/
      inline void compute(const Eigen::Matrix<var,R,C> &A) {
        Eigen::Matrix<double,R,C> Ad(A.rows(),A.cols());

        N_ = A.rows();
        _variA.resize(A.rows(),A.cols());

        for (size_t j = 0; j < N_; j++) {
          for (size_t i = 0; i < N_; i++) {
            Ad(i,j) = A(i,j).val();
            _variA(i,j) = A(i,j).vi_;
          }
        }
          
        _ldlt.compute(Ad);
      }

      /// Compute the log(abs(det(A))).  This is just a convenience function.
      inline double log_abs_det() const {
        return _ldlt.vectorD().array().log().sum();
      }
        
      size_t N_;
      Eigen::LDLT<Eigen::Matrix<double,R,C> > _ldlt;
      Eigen::Matrix<vari*,R,C> _variA;
    };
  }
}
#endif
