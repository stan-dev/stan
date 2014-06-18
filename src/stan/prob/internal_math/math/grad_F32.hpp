#ifndef __STAN__PROB__INTERNAL_MATH__MATH__GRAD_F32_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__GRAD_F32_HPP__

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
    void gradF32(T* g, T a, T b, T c, T d, T e, T z, T precision = 1e-6)
    {
      using std::log;
      using std::exp;
      using stan::agrad::log;
      using stan::agrad::exp;
      using stan::agrad::fabs;

      T gOld[6];
          
      for(T *p = g; p != g + 6; ++p) *p = 0;
      for(T *p = gOld; p != gOld + 6; ++p) *p = 0;
          
      T tOld = 1;
      T tNew = 0;
          
      T logT = 0;
          
      T logZ = log(z);
          
      int k = 0;
          
      while( (fabs(tNew) > precision) || (k == 0) )
        {


          T C = (a + k) / (d + k);
          C *= (b + k) / (e + k);
          C *= (c + k) / (1 + k);
              
          // If a, b, or c is a negative integer then the series terminates
          // after a finite number of interations
          if(C == 0) break;
              
          logT += (C > 0 ? 1 : -1) * log(fabs(C)) + logZ;
              
          tNew = exp(logT);
              
          gOld[0] = tNew * (gOld[0] / tOld + 1.0 / (a + k) );
          gOld[1] = tNew * (gOld[1] / tOld + 1.0 / (b + k) );
          gOld[2] = tNew * (gOld[2] / tOld + 1.0 / (c + k) );
              
          gOld[3] = tNew * (gOld[3] / tOld - 1.0 / (d + k) );
          gOld[4] = tNew * (gOld[4] / tOld - 1.0 / (e + k) );
              
          gOld[5] = tNew * ( gOld[5] / tOld + 1.0 / z );
              
          for(int i = 0; i < 6; ++i) g[i] += gOld[i];
              
          tOld = tNew;
          
          ++k;
              
        }
          
    }
      
      
  }

}

#endif
