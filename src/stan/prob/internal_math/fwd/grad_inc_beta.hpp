#ifndef STAN__PROB__INTERNAL_MATH__FWD__GRAD_INC_BETA_HPP
#define STAN__PROB__INTERNAL_MATH__FWD__GRAD_INC_BETA_HPP

#include <math.h>
#include <stan/agrad/fwd/functions/exp.hpp>
#include <stan/agrad/fwd/functions/log.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/functions/log.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/agrad/fwd/functions/value_of.hpp>
#include <stan/math/functions/value_of.hpp>
#include <stan/agrad/fwd/fvar.hpp>

#include <stan/prob/internal_math/fwd/inc_beta.hpp>
#include <stan/prob/internal_math/math/grad_2F1.hpp>

namespace stan {
    
  namespace agrad {
      
    // Gradient of the incomplete beta function beta(a, b, z)
    // with respect to the first two arguments, using the
    // equivalence to a hypergeometric function.
    // See http://dlmf.nist.gov/8.17#ii
    template<typename T>
    void grad_inc_beta(stan::agrad::fvar<T>& g1, 
                       stan::agrad::fvar<T>& g2, 
                       stan::agrad::fvar<T> a, 
                       stan::agrad::fvar<T> b, 
                       stan::agrad::fvar<T> z)
    {
      using stan::agrad::value_of;
      using stan::math::value_of;

      stan::agrad::fvar<T> c1 = log(z);
      stan::agrad::fvar<T> c2 = log(1 - z);
      stan::agrad::fvar<T> c3 = inc_beta(a, b, z);
          
      stan::agrad::fvar<T> C = exp( a * c1 + b * c2 ) / a;
          
      stan::agrad::fvar<T> dF1 = 0;
      stan::agrad::fvar<T> dF2 = 0;
          
      if(value_of(value_of(C))) stan::math::grad_2F1(dF1, dF2, a + b, 
                                                    (stan::agrad::fvar<T>)1.0, 
                                                    a + 1, z);

          
      g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
      g2 = c2 * c3 + C * dF1;
          
    }
      
  }

}

#endif
