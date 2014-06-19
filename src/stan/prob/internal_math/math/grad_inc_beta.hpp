#ifndef __STAN__PROB__INTERNAL_MATH__MATH__GRAD_INC_BETA_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__GRAD_INC_BETA_HPP__

#include <math.h>
#include <stan/prob/internal_math/math/inc_beta.hpp>
#include <stan/prob/internal_math/math/grad_2F1.hpp>

namespace stan {
    
  namespace math {
      
    // Gradient of the incomplete beta function beta(a, b, z)
    // with respect to the first two arguments, using the
    // equivalence to a hypergeometric function.
    // See http://dlmf.nist.gov/8.17#ii
    void grad_inc_beta(double& g1, double& g2, double a, double b, double z)
    {
      double c1 = std::log(z);
      double c2 = std::log(1 - z);
      double c3 = stan::math::inc_beta(a, b, z);
          
      double C = std::exp( a * c1 + b * c2 ) / a;
          
      double dF1 = 0;
      double dF2 = 0;
          
      if(C) stan::math::grad2F1(dF1, dF2, a + b, 1.0, a + 1, z);

          
      g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
      g2 = c2 * c3 + C * dF1;
          
    }  
      
  }

}

#endif
