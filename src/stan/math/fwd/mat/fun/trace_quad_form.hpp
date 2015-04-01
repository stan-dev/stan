#ifndef STAN__MATH__FWD__MAT__FUN__TRACE_QUAD_FORM_HPP
#define STAN__MATH__FWD__MAT__FUN__TRACE_QUAD_FORM_HPP

#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/fwd/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/transpose.hpp>
#include <stan/math/prim/mat/fun/trace.hpp>
#include <stan/math/fwd/core.hpp>

namespace stan {
  namespace agrad {

    template<int RA, int CA, int RB, int CB, typename T>
    inline stan::agrad::fvar<T>
    trace_quad_form(const Eigen::Matrix<stan::agrad::fvar<T>, RA, CA> &A,
                    const Eigen::Matrix<stan::agrad::fvar<T>, RB, CB> &B) {
      using stan::agrad::multiply;
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

