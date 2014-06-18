#ifndef __STAN__PROB__INTERNAL_MATH__MATH__F32_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__F32_HPP__

#include <math.h>
#include <stan/agrad/fwd/functions/exp.hpp>
#include <stan/agrad/fwd/functions/log.hpp>
#include <stan/agrad/fwd/functions/fabs.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/functions/log.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/rev/operators.hpp>

namespace stan {
    
  namespace math {

    template<typename T>
    T F32(T a, T b, T c, T d, T e, T z, T precision = 1e-6)
    {
      using std::exp;
      using std::log;
      using stan::agrad::exp;
      using stan::agrad::log;
      using stan::agrad::fabs;

      T F = 1;
          
      T tNew = 0;
          
      T logT = 0;
          
      T logZ = log(z);
          
      int k = 0;
          
      while( (fabs(tNew) > precision) || (k == 0) )
        {
              
          T p = (a + k) * (b + k) * (c + k) / ( (d + k) * (e + k) * (k + 1) );
              
          // If a, b, or c is a negative integer then the series terminates
          // after a finite number of interations
          if(p == 0) break;
              
          logT += (p > 0 ? 1 : -1) * log(fabs(p)) + logZ;
              
          tNew = exp(logT);
              
          F += tNew;
              
          ++k;

        }
          
      return F;
          
    }
      
  }

}

#endif
