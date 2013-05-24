#ifndef __STAN__MATH__MATRIX__TRACE_GEN_QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__TRACE_GEN_QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    /**
     * Compute trace(D B^T A B).
     **/
    template<int RD,int CD,int RA,int CA,int RB,int CB>
    inline double
    trace_gen_quad_form(const Eigen::Matrix<double,RD,CD> &D,
                        const Eigen::Matrix<double,RA,CA> &A,
                        const Eigen::Matrix<double,RB,CB> &B)
    {
      validate_square(A,"trace_gen_quad_form");
      validate_square(D,"trace_gen_quad_form");
      validate_multiplicable(A,B,"trace_gen_quad_form");
      validate_multiplicable(B,D,"trace_gen_quad_form");
      return (D*B.transpose()*A*B).trace();
    }
  }
}

#endif

