#ifndef STAN__MATH__MATRIX__QUAD_FORM_HPP
#define STAN__MATH__MATRIX__QUAD_FORM_HPP

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/error_handling/matrix/check_multiplicable.hpp>
#include <stan/error_handling/matrix/check_square.hpp>
#include <stan/error_handling/matrix/check_symmetric.hpp>
#include <stan/math/matrix/dot_product.hpp>
#include <stan/math/matrix/multiply.hpp>
#include <stan/math/matrix/transpose.hpp>

namespace stan {
  namespace math {
    /**
     * Compute B^T A B
     **/
    template<int RA,int CA,int RB,int CB,typename T>
    inline Eigen::Matrix<T,CB,CB>
    quad_form(const Eigen::Matrix<T,RA,CA>& A,
              const Eigen::Matrix<T,RB,CB>& B)
    {
      using stan::math::multiply;
      stan::error_handling::check_square("quad_form", "A", A);
      stan::error_handling::check_multiplicable("quad_form",
                                                "A", A,
                                                "B", B);
      return multiply(stan::math::transpose(B),multiply(A,B));
    }
    
    template<int RA,int CA,int RB,typename T>
    inline T
    quad_form(const Eigen::Matrix<T,RA,CA>& A,
              const Eigen::Matrix<T,RB,1>& B)
    {
      using stan::math::multiply;
      using stan::math::dot_product;

      stan::error_handling::check_square("quad_form", "A", A);
      stan::error_handling::check_multiplicable("quad_form",
                                                "A", A,
                                                "B", B);
      return dot_product(B,multiply(A,B));
    }
    
    template<int RA,int CA,int RB,int CB,typename T>
    inline Eigen::Matrix<T,CB,CB>
    quad_form_sym(const Eigen::Matrix<T,RA,CA>& A,
                  const Eigen::Matrix<T,RB,CB>& B)
    {
      using stan::math::multiply;
      
      stan::error_handling::check_square("quad_form_sym", "A", A);
      stan::error_handling::check_multiplicable("quad_form_sym", 
                                                "A", A, 
                                                "B", B);
      stan::error_handling::check_symmetric("quad_form_sym", "A", A);
      Eigen::Matrix<T,CB,CB> ret(multiply(stan::math::transpose(B),multiply(A,B)));
      return 0.5*(ret + stan::math::transpose(ret));
    }
    
    template<int RA,int CA,int RB,typename T>
    inline T
    quad_form_sym(const Eigen::Matrix<T,RA,CA>& A,
                  const Eigen::Matrix<T,RB,1>& B)
    {
      using stan::math::multiply;
      using stan::math::dot_product;

      stan::error_handling::check_square("quad_form_sym", "A", A);
      stan::error_handling::check_multiplicable("quad_form_sym", 
                                                "A", A, 
                                                "B", B);
      stan::error_handling::check_symmetric("quad_form_sym", "A", A);
      return dot_product(B,multiply(A,B));
    }
  }
}

#endif

