#ifndef STAN__AGRAD__FWD__MATRIX__TRACE_QUAD_FORM_HPP
#define STAN__AGRAD__FWD__MATRIX__TRACE_QUAD_FORM_HPP

#include <boost/type_traits.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/agrad/fwd/fvar.hpp>

namespace stan {
  namespace agrad {


    template<int RA,int CA,int RB,int CB,typename T>
    inline stan::agrad::fvar<T>
    trace_quad_form(const Eigen::Matrix<stan::agrad::fvar<T>,RA,CA> &A,
                    const Eigen::Matrix<stan::agrad::fvar<T>,RB,CB> &B)
    {
      using stan::agrad::multiply;
      using stan::math::multiply;
      stan::error_handling::check_square("trace_quad_form", "A", A);
      stan::error_handling::check_multiplicable("trace_quad_form", 
                                                "A", A, 
                                                "B", B);
      return stan::math::trace(multiply(stan::math::transpose(B),
                                        multiply(A,B)));
    }
  }
}

#endif

