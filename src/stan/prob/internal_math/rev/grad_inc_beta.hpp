#ifndef STAN__PROB__INTERNAL_MATH__REV__GRAD_INC_BETA_HPP
#define STAN__PROB__INTERNAL_MATH__REV__GRAD_INC_BETA_HPP

#include <math.h>
#include <stan/math/functions/lbeta.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/functions/lgamma.hpp>
#include <stan/agrad/rev/functions/log.hpp>
#include <stan/agrad/rev/functions/log1m.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/math/functions/value_of.hpp>

#include <stan/prob/internal_math/rev/inc_beta.hpp>
#include <stan/prob/internal_math/math/grad_2F1.hpp>

namespace stan {
    
  namespace agrad {
      
    // Gradient of the incomplete beta function beta(a, b, z)
    // with respect to the first two arguments, using the
    // equivalence to a hypergeometric function.
    // See http://dlmf.nist.gov/8.17#ii
    void grad_inc_beta(stan::agrad::var& g1, 
                       stan::agrad::var& g2, 
                       stan::agrad::var a, 
                       stan::agrad::var b, 
                       stan::agrad::var z) {
      using stan::agrad::value_of;
      using stan::math::value_of;

      stan::agrad::var c1 = stan::agrad::log(z);
      stan::agrad::var c2 = stan::agrad::log1m(z);
      stan::agrad::var c3 = stan::agrad::exp(stan::math::lbeta(a,b)) 
        * stan::agrad::inc_beta(a, b, z);
          
      stan::agrad::var C = exp( a * c1 + b * c2 ) / a;
          
      stan::agrad::var dF1 = 0;
      stan::agrad::var dF2 = 0;
          
      if(value_of(value_of(C))) stan::math::grad_2F1(dF1, dF2, a + b, 
                                                    (stan::agrad::var)1.0, 
                                                    a + 1, z);

          
      g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
      g2 = c2 * c3 + C * dF1;
          
    }
      
  }

}

#endif
