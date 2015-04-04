#ifndef STAN_MATH_PRIM_MAT_FUN_MDIVIDE_LEFT_TRI_LOW_HPP
#define STAN_MATH_PRIM_MAT_FUN_MDIVIDE_LEFT_TRI_LOW_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/fun/mdivide_left_tri.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
namespace stan {
  namespace math {

    template <typename T1, typename T2, int R1, int C1, int R2, int C2>
    inline
    Eigen::Matrix<typename boost::math::tools::promote_args<T1, T2>::type,
    R1, C2>
    mdivide_left_tri_low(const Eigen::Matrix<T1, R1, C1> &A,
                         const Eigen::Matrix<T2, R2, C2> &b) {
      stan::math::check_square("mdivide_left_tri_low", "A", A);
      stan::math::check_multiplicable("mdivide_left_tri_low",
                                                "A", A,
                                                "b", b);
      // return promote_common<Eigen::Matrix<T1, R1, C1>,
      //                       Eigen::Matrix<T2, R1, C1> >(A)
      //   .template triangularView<Eigen::Lower>()
      //   .solve( promote_common<Eigen::Matrix<T1, R2, C2>,
      //           Eigen::Matrix<T2, R2, C2> >(b) );
      return mdivide_left_tri<Eigen::Lower, T1, T2, R1, C1, R2, C2>(A, b);
    }
    template <typename T, int R1, int C1>
    inline
    Eigen::Matrix<T, R1, C1>
    mdivide_left_tri_low(const Eigen::Matrix<T, R1, C1> &A) {
      stan::math::check_square("mdivide_left_tri_low", "A", A);
      // int n = A.rows();
      // Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> b;
      // b.setIdentity(n, n);
      // A.template triangularView<Eigen::Lower>().solveInPlace(b);
      // return b;
      return mdivide_left_tri<Eigen::Lower, T, R1, C1>(A);
    }

  }
}
#endif
