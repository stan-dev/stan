#ifndef STAN__MATH__MATRIX__TRACE_INV_QUAD_FORM_LDLT_HPP
#define STAN__MATH__MATRIX__TRACE_INV_QUAD_FORM_LDLT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>

namespace stan {
  namespace math {
    
    /*
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(B^T A^-1 B)
     * where the LDLT_factor of A is provided.
     */
    template <int R2,int C2,int R3,int C3>
    inline double
    trace_inv_quad_form_ldlt(const stan::math::LDLT_factor<double,R2,C2> &A,
                             const Eigen::Matrix<double,R3,C3> &B) {
      stan::math::check_multiplicable("trace_inv_quad_form_ldlt(%1%)",A,"A",
                                      B,"B",(double*)0);
      
      return (B.transpose()*A._ldltP->solve(B)).trace();
    }
  }
}

#endif
