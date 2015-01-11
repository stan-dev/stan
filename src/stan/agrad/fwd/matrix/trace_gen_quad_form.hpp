#ifndef STAN__AGRAD__FWD__MATRIX__TRACE_GEN_QUAD_FORM_HPP
#define STAN__AGRAD__FWD__MATRIX__TRACE_GEN_QUAD_FORM_HPP

#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>

namespace stan {
  namespace agrad {
    template<int RD,int CD,int RA,int CA,int RB,int CB,typename T>
    inline fvar<T>
    trace_gen_quad_form(const Eigen::Matrix<fvar<T>,RD,CD> &D,
                        const Eigen::Matrix<fvar<T>,RA,CA> &A,
                        const Eigen::Matrix<fvar<T>,RB,CB> &B)
    {
      using stan::agrad::multiply;
      using stan::math::multiply;

      stan::error_handling::check_square("trace_gen_quad_form", "A", A);
      stan::error_handling::check_square("trace_gen_quad_form", "D", D);
      stan::error_handling::check_multiplicable("trace_gen_quad_form",
                                                "A", A,
                                                "B", B);
      stan::error_handling::check_multiplicable("trace_gen_quad_form", 
                                                "B", B, 
                                                "D", D);
      return stan::math::trace(multiply(multiply(D,stan::math::transpose(B)),
                                        multiply(A,B)));
    }
  }
}

#endif

