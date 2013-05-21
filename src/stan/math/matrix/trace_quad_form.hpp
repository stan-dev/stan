#ifndef __STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__TRACE_QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

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
      validate_square(A,"trace_quad_form");
      validate_multiplicable(A,B,"trace_quad_form");
      return (B.transpose()*A*B).trace();
    }
  }
}

#endif

