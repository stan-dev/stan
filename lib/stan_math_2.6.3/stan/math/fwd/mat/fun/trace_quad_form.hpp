#ifndef STAN_MATH_FWD_MAT_FUN_TRACE_QUAD_FORM_HPP
#define STAN_MATH_FWD_MAT_FUN_TRACE_QUAD_FORM_HPP

#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <stan/math/prim/mat/fun/trace.hpp>
#include <stan/math/fwd/core.hpp>

namespace stan {
  namespace math {

    template<int RA, int CA, int RB, int CB, typename T>
    inline stan::math::fvar<T>
    trace_quad_form(const Eigen::Matrix<stan::math::fvar<T>, RA, CA> &A,
                    const Eigen::Matrix<stan::math::fvar<T>, RB, CB> &B) {
      using stan::math::multiply;
      using stan::math::multiply;
      stan::math::check_square("trace_quad_form", "A", A);
      stan::math::check_multiplicable("trace_quad_form",
                                      "A", A,
                                      "B", B);
      return stan::math::trace(multiply(stan::math::transpose(B),
                                        multiply(A, B)));
    }
  }
}

#endif

