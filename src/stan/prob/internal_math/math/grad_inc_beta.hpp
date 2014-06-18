#ifndef __STAN__PROB__INTERNAL_MATH__MATH__GRAD_INC_BETA_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__GRAD_INC_BETA_HPP__

#include <math.h>
#include <stan/agrad/fwd/functions/exp.hpp>
#include <stan/agrad/fwd/functions/log.hpp>
#include <stan/agrad/fwd/functions/fabs.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/functions/log.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <stan/agrad/rev/functions/value_of.hpp>
#include <stan/agrad/fwd/functions/value_of.hpp>
#include <stan/math/functions/value_of.hpp>

#include <stan/prob/internal_math/fwd/beta.hpp>
#include <stan/prob/internal_math/math/grad_2F1.hpp>
namespace stan {
    
  namespace math {
      
    // Gradient of the incomplete beta function beta(a, b, z)
    // with respect to the first two arguments, using the
    // equivalence to a hypergeometric function.
    // See http://dlmf.nist.gov/8.17#ii
    template<typename T>
    void gradIncBeta(T& g1, T& g2, T a, T b, T z)
    {
      using std::log;
      using std::exp;
      using stan::agrad::value_of;
      using stan::math::value_of;
      using boost::math::beta;

      T c1 = log(z);
      T c2 = log(1 - z);
      T c3 = beta(a, b, z);
          
      T C = exp( a * c1 + b * c2 ) / a;
          
      T dF1 = 0;
      T dF2 = 0;
          
      if(value_of(value_of(C))) stan::math::grad2F1(dF1, dF2, a + b, (T)1.0, a + 1, z);

          
      g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
      g2 = c2 * c3 + C * dF1;
          
    }
      
  }

}

#endif
