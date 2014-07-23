#ifndef STAN__MATH__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP
#define STAN__MATH__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>

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
    
      stan::math::check_square("trace_gen_inv_quad_form_ldlt(%1%)",D,"D",
                               (double*)0);
      stan::math::check_multiplicable("trace_gen_inv_quad_form_ldlt(%1%)",A,"A",
                                      B,"B",(double*)0);
      stan::math::check_multiplicable("trace_gen_inv_quad_form_ldlt(%1%)",B,"B",
                                      D,"D",(double*)0);
      
      return (D*B.transpose()*A._ldltP->solve(B)).trace();
    }

  }
}
#endif
