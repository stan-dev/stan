#ifndef STAN__MATH__PRIM__MAT__FUN__TRACE_QUAD_FORM_HPP
#define STAN__MATH__PRIM__MAT__FUN__TRACE_QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/prim/mat/fun/Eigen.hpp>
#include <stan/math/prim/mat/err/check_multiplicable.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>

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
      stan::math::check_square("trace_quad_form", "A", A);
      stan::math::check_multiplicable("trace_quad_form",
                                                "A", A,
                                                "B", B);

      return (B.transpose()*A*B).trace();
    }

  }
}

#endif

