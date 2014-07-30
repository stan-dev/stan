#ifndef STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP
#define STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>

namespace stan {
  namespace math {
    /**
     * Compute trace(B^T A B).
     **/
    template<int RA,int CA,int RB,int CB>
    inline double
    trace_quad_form(const Eigen::Matrix<double,RA,CA> &A,
                    const Eigen::Matrix<double,RB,CB> &B)
    {
      stan::math::check_square("trace_quad_form(%1%)",A,"A",(double*)0);
      stan::math::check_multiplicable("trace_quad_form(%1%)",A,"A",
                                      B,"B",(double*)0);

      return (B.transpose()*A*B).trace();
    }

  }
}

#endif

