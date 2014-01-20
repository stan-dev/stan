#ifndef __STAN__MATH__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP__
#define __STAN__MATH__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP__

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>

namespace stan {
  namespace math {
    
    /*
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(D B^T A^-1 B)
     * where D is a square matrix and the LDLT_factor of A is provided.
     */
    template <int R1,int C1,int R2,int C2,int R3,int C3>
    inline double
    trace_gen_inv_quad_form_ldlt(const Eigen::Matrix<double,R1,C1> &D,
                                 const stan::math::LDLT_factor<double,R2,C2> &A,
                                 const Eigen::Matrix<double,R3,C3> &B) {
    
      stan::math::validate_square(D,"trace_gen_inv_quad_form_ldlt");
      stan::math::validate_multiplicable(A,B,"trace_gen_inv_quad_form_ldlt");
      stan::math::validate_multiplicable(B,D,"trace_gen_inv_quad_form_ldlt");
      
      return (D*B.transpose()*A._ldltP->solve(B)).trace();
    }

  }
}
#endif
