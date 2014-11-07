#ifndef STAN__MATH__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP
#define STAN__MATH__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/LDLT_factor.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/math/matrix/mdivide_left_ldlt.hpp>
#include <stan/math/matrix/trace.hpp>
#include <stan/math/matrix/transpose.hpp>
#include <stan/math/matrix/multiply.hpp>

namespace stan {
  namespace math {
    
    /*
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(D B^T A^-1 B)
     * where D is a square matrix and the LDLT_factor of A is provided.
     */
    template <typename T1, typename T2, typename T3,
              int R1,int C1,int R2,int C2,int R3,int C3>
    inline typename 
    boost::enable_if_c<!stan::is_var<T1>::value && 
                       !stan::is_var<T2>::value && 
                       !stan::is_var<T3>::value, 
                       typename boost::math::tools::promote_args<T1,T2,T3>::type>::type
    trace_gen_inv_quad_form_ldlt(const Eigen::Matrix<T1,R1,C1> &D,
                                 const stan::math::LDLT_factor<T2,R2,C2> &A,
                                 const Eigen::Matrix<T3,R3,C3> &B) {
    
      stan::error_handling::check_square("trace_gen_inv_quad_form_ldlt", "D", D);
      stan::error_handling::check_multiplicable("trace_gen_inv_quad_form_ldlt", 
                                                "A", A,
                                                "B", B);
      stan::error_handling::check_multiplicable("trace_gen_inv_quad_form_ldlt",
                                                "B", B,
                                                "D", D);
      
      return trace(multiply(multiply(D,transpose(B)),mdivide_left_ldlt(A,B)));
    }

  }
}
#endif
