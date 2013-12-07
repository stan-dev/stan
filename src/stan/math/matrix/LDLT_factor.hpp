#ifndef __STAN__MATH__MATRIX__LDLT_FACTOR_HPP__
#define __STAN__MATH__MATRIX__LDLT_FACTOR_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <boost/shared_ptr.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {

    // This class is conceptually similar to the corresponding Eigen class
    // Any spd matrix A can be decomposed as LDL' where L is unit lower-triangular
    // and D is diagonal with positive diagonal elements

    template<typename T, int R, int C>
    class LDLT_factor;
    
    template<int R, int C>
    class LDLT_factor<double,R,C> {
    public:
      LDLT_factor()
      : N_(0), _ldltP(new Eigen::LDLT< Eigen::Matrix<double,R,C> >()) {}

      LDLT_factor(const Eigen::Matrix<double,R,C> &A)
      : N_(0), _ldltP(new Eigen::LDLT< Eigen::Matrix<double,R,C> >())
      {
        compute(A);
      }
      
      inline void compute(const Eigen::Matrix<double,R,C> &A) {
        stan::math::validate_square(A,"LDLT_factor<double>::compute");
        N_ = A.rows();
        _ldltP->compute(A);
      }
      
      inline bool success() const {
        bool ret;
        ret = _ldltP->info() == Eigen::Success;
        ret = ret && _ldltP->isPositive();
        ret = ret && (_ldltP->vectorD().array() > 0).all();
        return ret;
      }

      inline double log_abs_det() const {
        return _ldltP->vectorD().array().log().sum();
      }
      
      inline void inverse(Eigen::Matrix<double,R,C> &invA) const {
        invA.setIdentity(N_);
        _ldltP->solveInPlace(invA);
      }

      template<typename Rhs>
      inline const Eigen::internal::solve_retval<Eigen::LDLT< Eigen::Matrix<double,R,C> >, Rhs>
      solve(const Eigen::MatrixBase<Rhs>& b) const {
        return _ldltP->solve(b);
      }

      inline Eigen::Matrix<double,R,C> solveRight(const Eigen::Matrix<double,R,C> &B) const {
        return _ldltP->solve(B.transpose()).transpose();
      }
      
      inline size_t rows() const { return N_; }
      inline size_t cols() const { return N_; }
      
      typedef size_t size_type;

      size_t N_;
      boost::shared_ptr< Eigen::LDLT< Eigen::Matrix<double,R,C> > > _ldltP;
    };
  }
}
#endif
