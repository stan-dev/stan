#ifndef __STAN__MATH__MATRIX__LDLT_HPP__
#define __STAN__MATH__MATRIX__LDLT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

#include <boost/shared_ptr.hpp>
#include <boost/math/tools/promotion.hpp>

namespace stan {
  namespace math {
    
    template<typename T, int R, int C>
    class LDLT_factor;
    
    template<int R, int C>
    class LDLT_factor<double,R,C> {
    public:
      LDLT_factor() : N_(0) {}
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
    
    template <int R1,int C1,int R2,int C2>
    inline Eigen::Matrix<double,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<double,R1,C1> &A,
                      const Eigen::Matrix<double,R2,C2> &b) {
      stan::math::validate_multiplicable(A,b,"mdivide_left_ldlt");
      
      return A.solve(b);
    }
    
    template <typename T1, typename T2, int R1, int C1, int R2, int C2>
    inline 
    Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R1,C2>
    mdivide_right_ldlt(const Eigen::Matrix<T1,R1,C1> &b,
                       const stan::math::LDLT_factor<T2,R2,C2> &A) {
      stan::math::validate_multiplicable(b,A,"mdivide_right_ldlt");

      return transpose(mdivide_left_ldlt(A,transpose(b)));
    }
    
    template <int R1, int C1, int R2, int C2>
    inline Eigen::Matrix<double,R1,C2>
    mdivide_right_ldlt(const Eigen::Matrix<double,R1,C1> &b,
                       const stan::math::LDLT_factor<double,R2,C2> &A) {
      stan::math::validate_multiplicable(b,A,"mdivide_right_ldlt");

      return A.solveRight(b);
    }
    
    template<int R, int C>
    inline double
    log_determinant_ldlt(stan::math::LDLT_factor<double,R,C> &A) {
      return A.log_abs_det();
    }
    
    /*
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(D B^T A^-1 B)
     * where D is a square matrix and the LDLT_factor of A is provided.
     */
    template <int R1,int C1,int R2,int C2,int R3,int C3>
    inline double
    trace_inv_quad_form_ldlt(const Eigen::Matrix<double,R1,C1> &D,
                             const stan::math::LDLT_factor<double,R2,C2> &A,
                             const Eigen::Matrix<double,R3,C3> &B)
    {
      stan::math::validate_square(D,"trace_inv_quad_form_ldlt");
      stan::math::validate_multiplicable(A,B,"trace_inv_quad_form_ldlt");
      stan::math::validate_multiplicable(B,D,"trace_inv_quad_form_ldlt");
      
      return (D*B.transpose()*A._ldltP->solve(B)).trace();
    }

    /*
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(B^T A^-1 B)
     * where the LDLT_factor of A is provided.
     */
    template <int R2,int C2,int R3,int C3>
    inline double
    trace_inv_quad_form_ldlt(const stan::math::LDLT_factor<double,R2,C2> &A,
                             const Eigen::Matrix<double,R3,C3> &B)
    {
      stan::math::validate_multiplicable(A,B,"trace_inv_quad_form_ldlt");
      
      return (B.transpose()*A._ldltP->solve(B)).trace();
    }
  }
}
#endif
