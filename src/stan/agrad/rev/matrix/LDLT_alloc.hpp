#ifndef __STAN__AGRAD__REV__MATRIX__LDLT_ALLOC_HPP__
#define __STAN__AGRAD__REV__MATRIX__LDLT_ALLOC_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>

namespace stan {
  namespace agrad {
    template<int R, int C>
    class LDLT_alloc : public chainable_alloc {
    public:
      LDLT_alloc() : N_(0) {}
      LDLT_alloc(const Eigen::Matrix<var,R,C> &A) : N_(0) {
        compute(A);
      }
        
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
      inline double log_abs_det() const {
        return _ldlt.vectorD().array().log().sum();
      }
        
      size_t N_;
      Eigen::LDLT< Eigen::Matrix<double,R,C> > _ldlt;
      Eigen::Matrix<vari*,R,C> _variA;
    };
  }
}
#endif
