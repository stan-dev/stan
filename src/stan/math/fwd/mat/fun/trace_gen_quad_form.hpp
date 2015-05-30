#ifndef STAN_MATH_FWD_MAT_FUN_TRACE_GEN_QUAD_FORM_HPP
#define STAN_MATH_FWD_MAT_FUN_TRACE_GEN_QUAD_FORM_HPP

#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/trace.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>

namespace stan {
  namespace math {
    template<int RD, int CD, int RA, int CA, int RB, int CB, typename T>
    inline fvar<T>
    trace_gen_quad_form(const Eigen::Matrix<fvar<T>, RD, CD> &D,
                        const Eigen::Matrix<fvar<T>, RA, CA> &A,
                        const Eigen::Matrix<fvar<T>, RB, CB> &B) {
      using stan::math::multiply;
      using stan::math::multiply;

      stan::math::check_square("trace_gen_quad_form", "A", A);
      stan::math::check_square("trace_gen_quad_form", "D", D);
      stan::math::check_multiplicable("trace_gen_quad_form",
                                      "A", A,
                                      "B", B);
      stan::math::check_multiplicable("trace_gen_quad_form",
                                      "B", B,
                                      "D", D);
      return stan::math::trace(multiply(multiply(D, stan::math::transpose(B)),
                                        multiply(A, B)));
    }
  }
}

#endif

