#ifndef STAN__MATH__MATRIX__MDIVIDE_LEFT_LDLT_HPP
#define STAN__MATH__MATRIX__MDIVIDE_LEFT_LDLT_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/matrix/promote_common.hpp>
#include <boost/type_traits/is_same.hpp>

namespace stan {
  namespace math {

    /**
     * Returns the solution of the system Ax=b given an LDLT_factor of A
     * @param A LDLT_factor
     * @param b Right hand side matrix or vector.
     * @return x = b A^-1, solution of the linear system.
     * @throws std::domain_error if rows of b don't match the size of A.
     */

    template <int R1,int C1,int R2,int C2, typename T1, typename T2>
    inline Eigen::Matrix<typename boost::math::tools::promote_args<T1,T2>::type,R1,C2>
    mdivide_left_ldlt(const stan::math::LDLT_factor<T1,R1,C1> &A,
                      const Eigen::Matrix<T2,R2,C2> &b) {
      stan::error_handling::check_multiplicable("mdivide_left_ldlt", 
                                                "A", A,
                                                "b", b);
      
      return A.solve(promote_common<Eigen::Matrix<T1,R2,C2>,
                                      Eigen::Matrix<T2,R2,C2> >(b));
    }

  }
}
#endif
