#ifndef STAN__AGRAD__REV__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP
#define STAN__AGRAD__REV__MATRIX__TRACE_GEN_INV_QUAD_FORM_LDLT_HPP

#include <stan/math/matrix/Eigen.hpp>
#include <stan/agrad/rev/var.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/agrad/rev/matrix/trace_inv_quad_form_ldlt.hpp>

namespace stan {
  namespace agrad {

    /**
     * Compute the trace of an inverse quadratic form.  I.E., this computes
     *       trace(D B^T A^-1 B)
     * where D is a square matrix and the LDLT_factor of A is provided.
     **/
    template <typename T1,int R1,int C1,typename T2,int R2,int C2,typename T3,int R3,int C3>
    inline typename
    boost::enable_if_c<boost::is_same<T1,var>::value || 
    boost::is_same<T2,var>::value || 
                       boost::is_same<T3,var>::value, var>::type
    trace_gen_inv_quad_form_ldlt(const Eigen::Matrix<T1,R1,C1> &D,
                                 const stan::math::LDLT_factor<T2,R2,C2> &A,
                                 const Eigen::Matrix<T3,R3,C3> &B)
    {
      stan::math::check_square("trace_gen_inv_quad_form_ldlt(%1%)",D,"D",
                               (double*)0);
      stan::math::check_multiplicable("trace_gen_inv_quad_form_ldlt(%1%)",A,"A",
                                      B,"B",(double*)0);
      stan::math::check_multiplicable("trace_gen_inv_quad_form_ldlt(%1%)",B,"B",
                                      D,"D",(double*)0);
      
      trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3> *_impl = new trace_inv_quad_form_ldlt_impl<T2,R2,C2,T3,R3,C3>(D,A,B);
      
      return var(new trace_inv_quad_form_ldlt_vari<T2,R2,C2,T3,R3,C3>(_impl));
    }


  }
}

#endif
