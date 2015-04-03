#ifndef STAN_MATH_PRIM_MAT_FUN_TRACE_GEN_QUAD_FORM_HPP
#define STAN_MATH_PRIM_MAT_FUN_TRACE_GEN_QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

namespace stan {
  namespace math {
    /**
     * Compute trace(D B^T A B).
     **/
    template<int RD, int CD, int RA, int CA, int RB, int CB>
    inline double
    trace_gen_quad_form(const Eigen::Matrix<double, RD, CD> &D,
                        const Eigen::Matrix<double, RA, CA> &A,
                        const Eigen::Matrix<double, RB, CB> &B) {
      stan::math::check_square("trace_gen_quad_form", "A", A);
      stan::math::check_square("trace_gen_quad_form", "D", D);
      stan::math::check_multiplicable("trace_gen_quad_form",
                                      "A", A,
                                      "B", B);
      stan::math::check_multiplicable("trace_gen_quad_form",
                                      "B", B,
                                      "D", D);
      return (D*B.transpose()*A*B).trace();
    }
  }
}

#endif

