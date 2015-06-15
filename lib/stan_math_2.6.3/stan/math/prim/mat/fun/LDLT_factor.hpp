#ifndef STAN_MATH_PRIM_MAT_FUN_LDLT_FACTOR_HPP
#define STAN_MATH_PRIM_MAT_FUN_LDLT_FACTOR_HPP

#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <boost/shared_ptr.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/scal/fun/is_nan.hpp>

namespace stan {

  namespace math {

    // This class is conceptually similar to the corresponding Eigen class
    // Any spd matrix A can be decomposed as LDL' where L is unit
    // lower-triangular and D is diagonal with positive diagonal elements

    template<typename T, int R, int C>
    class LDLT_factor;

    /**
     * LDLT_factor is a thin wrapper on Eigen::LDLT to allow for
     * reusing factorizations and efficient autodiff of things like
     * log determinants and solutions to linear systems.
     *
     * After the constructor and/or compute() is called users of
     * LDLT_factor are responsible for calling success() to
     * check whether the factorization has succeeded.  Use of an LDLT_factor
     * object (e.g., in mdivide_left_ldlt) is undefined if success() is false.
     *
     * It's usage pattern is:
     *
     * ~~~
     * Eigen::Matrix<T, R, C> A1, A2;
     *
     * LDLT_factor<T, R, C> ldlt_A1(A1);
     * LDLT_factor<T, R, C> ldlt_A2;
     * ldlt_A2.compute(A2);
     * ~~~
     *
     * Now, the caller should check that ldlt_A1.success() and ldlt_A2.success()
     * are true or abort accordingly.  Alternatively, call check_ldlt_factor().
     *
     * Note that ldlt_A1 and ldlt_A2 are completely equivalent.  They simply
     * demonstrate two different ways to construct the factorization.
     *
     * Now, the caller can use the LDLT_factor objects as needed.  For instance
     *
     * ~~~
     * x1 = mdivide_left_ldlt(ldlt_A1, b1);
     * x2 = mdivide_right_ldlt(b2, ldlt_A2);
     *
     * d1 = log_determinant_ldlt(ldlt_A1);
     * d2 = log_determinant_ldlt(ldlt_A2);
     * ~~~
     *
     **/
    template<int R, int C, typename T>
    class LDLT_factor<T, R, C> {
    public:
      LDLT_factor()
        : N_(0), _ldltP(new Eigen::LDLT< Eigen::Matrix<T, R, C> >()) {}

      explicit LDLT_factor(const Eigen::Matrix<T, R, C> &A)
        : N_(0), _ldltP(new Eigen::LDLT< Eigen::Matrix<T, R, C> >()) {
        compute(A);
      }

      inline void compute(const Eigen::Matrix<T, R, C> &A) {
        stan::math::check_square("LDLT_factor", "A", A);
        N_ = A.rows();
        _ldltP->compute(A);
      }

      inline bool success() const {
        using stan::math::is_nan;
        // bool ret;
        // ret = _ldltP->info() == Eigen::Success;
        // ret = ret && _ldltP->isPositive();
        // ret = ret && (_ldltP->vectorD().array() > 0).all();
        // return ret;

        if (_ldltP->info() != Eigen::Success)
          return false;
        if (!(_ldltP->isPositive()))
          return false;
        Eigen::Matrix<T, Eigen::Dynamic, 1> ldltP_diag(_ldltP->vectorD());
        for (int i = 0; i < ldltP_diag.size(); ++i)
          if (ldltP_diag(i) <= 0 || is_nan(ldltP_diag(i)))
            return false;
        return true;
      }

      inline T log_abs_det() const {
        return _ldltP->vectorD().array().log().sum();
      }

      inline void inverse(Eigen::Matrix<T, R, C> &invA) const {
        invA.setIdentity(N_);
        _ldltP->solveInPlace(invA);
      }

      template<typename Rhs>
      inline const
      Eigen::internal::solve_retval<Eigen::LDLT< Eigen::Matrix<T, R, C> >, Rhs>
      solve(const Eigen::MatrixBase<Rhs>& b) const {
        return _ldltP->solve(b);
      }

      inline Eigen::Matrix<T, R, C>
      solveRight(const Eigen::Matrix<T, R, C> &B) const {
        return _ldltP->solve(B.transpose()).transpose();
      }

      inline Eigen::Matrix<T, Eigen::Dynamic, 1> vectorD() const {
        return _ldltP->vectorD();
      }

      inline Eigen::LDLT<Eigen::Matrix<T, R, C> > matrixLDLT() const {
        return _ldltP->matrixLDLT();
      }

      inline size_t rows() const { return N_; }
      inline size_t cols() const { return N_; }

      typedef size_t size_type;
      typedef double value_type;

      size_t N_;
      boost::shared_ptr< Eigen::LDLT< Eigen::Matrix<T, R, C> > > _ldltP;
    };
  }
}
#endif
