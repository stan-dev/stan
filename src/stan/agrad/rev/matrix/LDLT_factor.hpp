#ifndef __STAN__AGRAD__REV__MATRIX__LDLT_FACTOR_HPP__
#define __STAN__AGRAD__REV__MATRIX__LDLT_FACTOR_HPP__

#include <stan/agrad/rev/var.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/agrad/rev/matrix/LDLT_alloc.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>

namespace stan {
  namespace math {
    // template specialization for src/stan/math/matrix/LDLT_factor.hpp
    template<int R, int C>
    class LDLT_factor<stan::agrad::var,R,C> {
    public:
      LDLT_factor() : _alloc(new stan::agrad::LDLT_alloc<R,C>()) {}
      LDLT_factor(const Eigen::Matrix<stan::agrad::var,R,C> &A) : _alloc(new stan::agrad::LDLT_alloc<R,C>(A)) { }
      
      inline void compute(const Eigen::Matrix<stan::agrad::var,R,C> &A) {
        stan::math::validate_square(A,"LDLT_factor<var>::compute");
        _alloc->compute(A);
      }
      
      template<typename Rhs>
      inline const Eigen::internal::solve_retval<Eigen::LDLT< Eigen::Matrix<double,R,C> >, Rhs>
      solve(const Eigen::MatrixBase<Rhs>& b) const {
        return _alloc->_ldlt.solve(b);
      }
      
      inline bool success() const {
        bool ret;
        ret = _alloc->_ldlt.info() == Eigen::Success;
        ret = ret && _alloc->_ldlt.isPositive();
        ret = ret && (_alloc->_ldlt.vectorD().array() > 0).all();
        return ret;
      }

      inline Eigen::VectorXd vectorD() const {
        return _alloc->_ldlt.vectorD();
      }

      inline size_t rows() const { return _alloc->N_; }
      inline size_t cols() const { return _alloc->N_; }
      
      typedef size_t size_type;

      stan::agrad::LDLT_alloc<R,C> *_alloc;
    };
  }
}
#endif
