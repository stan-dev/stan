#ifndef STAN_MATH_PRIM_MAT_FUN_QUAD_FORM_SYM_HPP
#define STAN_MATH_PRIM_MAT_FUN_QUAD_FORM_SYM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>

namespace stan {
  namespace math {

    template<int RA, int CA, int RB, int CB, typename T>
    inline Eigen::Matrix<T, CB, CB>
    quad_form_sym(const Eigen::Matrix<T, RA, CA>& A,
                  const Eigen::Matrix<T, RB, CB>& B) {
      using stan::math::multiply;

      stan::math::check_square("quad_form_sym", "A", A);
      stan::math::check_multiplicable("quad_form_sym",
                                                "A", A,
                                                "B", B);
      stan::math::check_symmetric("quad_form_sym", "A", A);
      Eigen::Matrix<T, CB, CB> ret(multiply(transpose(B), multiply(A, B)));
      return T(0.5) * (ret + transpose(ret));
    }

    template<int RA, int CA, int RB, typename T>
    inline T
    quad_form_sym(const Eigen::Matrix<T, RA, CA>& A,
                  const Eigen::Matrix<T, RB, 1>& B) {
      using stan::math::multiply;
      using stan::math::dot_product;

      stan::math::check_square("quad_form_sym", "A", A);
      stan::math::check_multiplicable("quad_form_sym",
                                                "A", A,
                                                "B", B);
      stan::math::check_symmetric("quad_form_sym", "A", A);
      return dot_product(B, multiply(A, B));
    }
  }
}

#endif

