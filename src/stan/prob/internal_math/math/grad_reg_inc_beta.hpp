#ifndef __STAN__PROB__INTERNAL_MATH__MATH__GRAD_REG_INC_BETA_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__GRAD_REG_INC_BETA_HPP__

#include <math.h>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <boost/math/special_functions/beta.hpp>

#include <stan/prob/internal_math/math/grad_inc_beta.hpp>

namespace stan {
    
  namespace math {

    // Gradient of the regularized incomplete beta function ibeta(a, b, z)
    template<typename T>
    void gradRegIncBeta(T& g1, T& g2, T a, T b, T z, 
                        T digammaA, T digammaB, T digammaSum, T betaAB)
    {
      using boost::math::beta;
      using stan::math::gradIncBeta;

      T dBda = 0;
      T dBdb = 0;
          
      gradIncBeta(dBda, dBdb, a, b, z);
          
      T b1 = beta(a, b, z);
          
      g1 = ( dBda - b1 * (digammaA - digammaSum) ) / betaAB;
      g2 = ( dBdb - b1 * (digammaB - digammaSum) ) / betaAB;
          
    }
      
      
  }

}

#endif
