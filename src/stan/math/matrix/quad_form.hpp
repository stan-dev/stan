#ifndef __STAN__MATH__MATRIX__QUAD_FORM_HPP__
#define __STAN__MATH__MATRIX__QUAD_FORM_HPP__

#include <boost/utility/enable_if.hpp>
#include <boost/type_traits.hpp>
#include <stan/math/matrix/Eigen.hpp>
#include <stan/math/matrix/validate_multiplicable.hpp>
#include <stan/math/matrix/validate_square.hpp>
#include <stan/math/matrix/validate_symmetric.hpp>

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
      validate_square(A,"quad_form");
      validate_multiplicable(A,B,"quad_form");
      return B.transpose()*A*B;
    }
    
    template<int RA,int CA,int RB>
    inline double
    quad_form(const Eigen::Matrix<double,RA,CA> &A,
              const Eigen::Matrix<double,RB,1> &B)
    {
      validate_square(A,"quad_form");
      validate_multiplicable(A,B,"quad_form");
      return B.dot(A*B);
    }
    
    template<int RA,int CA,int RB,int CB>
    inline Eigen::Matrix<double,CB,CB>
    quad_form_sym(const Eigen::Matrix<double,RA,CA> &A,
                  const Eigen::Matrix<double,RB,CB> &B)
    {
      validate_square(A,"quad_form_sym");
      validate_multiplicable(A,B,"quad_form_sym");
      validate_symmetric(A,"quad_form_sym");
      Eigen::Matrix<double,CB,CB> ret(B.transpose()*A*B);
      return 0.5*(ret + ret.transpose());
    }
    
    template<int RA,int CA,int RB>
    inline double
    quad_form_sym(const Eigen::Matrix<double,RA,CA> &A,
                  const Eigen::Matrix<double,RB,1> &B)
    {
      validate_square(A,"quad_form_sym");
      validate_multiplicable(A,B,"quad_form_sym");
      validate_symmetric(A,"quad_form_sym");
      return B.dot(A*B);
    }
  }
}

#endif

