#ifndef __STAN__MATH__MATRIX__QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/error_handling/matrix/check_multiplicable.hpp>
#include <stan/math/error_handling/matrix/check_square.hpp>
#include <stan/math/error_handling/matrix/check_symmetric.hpp>

namespace stan {
  namespace math {
    /**
     * Compute B^T A B
     **/
    template<int RA,int CA,int RB,int CB>
    inline Eigen::Matrix<double,CB,CB>
    quad_form(const Eigen::Matrix<double,RA,CA> &A,
              const Eigen::Matrix<double,RB,CB> &B)
    {
      stan::math::check_square("quad_form(%1%)",A,"A",(double*)0);
      stan::math::check_multiplicable("quad_form(%1%)",A,"A",
                                      B,"B",(double*)0);
      return B.transpose()*A*B;
    }
    
    template<int RA,int CA,int RB>
    inline double
    quad_form(const Eigen::Matrix<double,RA,CA> &A,
              const Eigen::Matrix<double,RB,1> &B)
    {
      stan::math::check_square("quad_form(%1%)",A,"A",(double*)0);
      stan::math::check_multiplicable("quad_form(%1%)",A,"A",
                                      B,"B",(double*)0);
      return B.dot(A*B);
    }
    
    template<int RA,int CA,int RB,int CB>
    inline Eigen::Matrix<double,CB,CB>
    quad_form_sym(const Eigen::Matrix<double,RA,CA> &A,
                  const Eigen::Matrix<double,RB,CB> &B)
    {
      stan::math::check_square("quad_form_sym(%1%)",A,"A",(double*)0);
      stan::math::check_multiplicable("quad_form_sym(%1%)",A,"A",
                                      B,"B",(double*)0);
      stan::math::check_symmetric("quad_form_sym(%1%)",A,"A",(double*)0);
      Eigen::Matrix<double,CB,CB> ret(B.transpose()*A*B);
      return 0.5*(ret + ret.transpose());
    }
    
    template<int RA,int CA,int RB>
    inline double
    quad_form_sym(const Eigen::Matrix<double,RA,CA> &A,
                  const Eigen::Matrix<double,RB,1> &B)
    {
      stan::math::check_square("quad_form_sym(%1%)",A,"A",(double*)0);
      stan::math::check_multiplicable("quad_form_sym(%1%)",A,"A",
                                      B,"B",(double*)0);    
      stan::math::check_symmetric("quad_form_sym(%1%)",A,"A",(double*)0);
      return B.dot(A*B);
    }
  }
}

#endif

