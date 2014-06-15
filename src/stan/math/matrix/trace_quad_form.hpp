#ifndef __STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/agrad/fwd/matrix/multiply.hpp>
#include <stan/agrad/rev/matrix/multiply.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/agrad/rev/operators.hpp>

namespace stan {
  namespace math {
    /**
     * Compute trace(B^T A B).
     **/
    template<int RA,int CA,int RB,int CB,typename T>
    inline T
    trace_quad_form(const Eigen::Matrix<T,RA,CA> &A,
                    const Eigen::Matrix<T,RB,CB> &B)
    {
      using stan::agrad::multiply;
      using stan::math::multiply;
      stan::math::check_square("trace_quad_form(%1%)",A,"A",(double*)0);
      stan::math::check_multiplicable("trace_quad_form(%1%)",A,"A",
                                      B,"B",(double*)0);
      return stan::math::trace(multiply(stan::math::transpose(B),
                                        multiply(A,B)));
    }
  }
}

#endif

