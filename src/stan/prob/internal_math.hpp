#ifndef __STAN__PROB__INTERNAL_MATH_HPP__
#define __STAN__PROB__INTERNAL_MATH_HPP__

#include <math.h>
#include <stan/agrad/fwd/functions/exp.hpp>
#include <stan/agrad/fwd/functions/log.hpp>
#include <stan/agrad/fwd/functions/fabs.hpp>
#include <stan/agrad/fwd/functions/gamma_p.hpp>
#include <stan/agrad/fwd/operators.hpp>
#include <stan/agrad/rev/functions/exp.hpp>
#include <stan/agrad/rev/functions/log.hpp>
#include <stan/agrad/rev/functions/fabs.hpp>
#include <stan/agrad/rev/functions/gamma_p.hpp>
#include <stan/agrad/rev/operators.hpp>
#include <stan/math/functions/gamma_p.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
    
  namespace math {

    template<typename T>
    T F32(T a, T b, T c, T d, T e, T z, T precision = 1e-6)
    {
      using std::exp;
      using std::log;
          
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
      
    template<typename T>
    void gradF32(T* g, T a, T b, T c, T d, T e, T z, T precision = 1e-6)
    {
      using std::log;
      using std::exp;
          
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
      
    // Gradient of the hypergeometric function 2F1(a, b | c | z) with respect to a and c
    template<typename T>
    void grad2F1(T& gradA, T& gradC, T a, T b, T c, T z, T precision = 1e-6)
    {
          
      gradA = 0;
      gradC = 0;
          
      T gradAold = 0;
      T gradCold = 0;
          
      int k = 0;
      T tDak = 1.0 / (a - 1);
          
      while( (fabs(tDak * (a + (k - 1)) ) > precision) || (k == 0) )
        {
              
          const T r = ( (a + k) / (c + k) ) * ( (b + k) / (T)(k + 1) ) * z;
          tDak = r * tDak * (a + (k - 1)) / (a + k);

          if(r == 0) break;
              
          gradAold = r * gradAold + tDak;
          gradCold = r * gradCold - tDak * ((a + k) / (c + k));
              
          gradA += gradAold;
          gradC += gradCold;
              
          ++k;
              
          if(k > 200) break;

        }
          
    }
      
    // Gradient of the incomplete beta function beta(a, b, z)
    // with respect to the first two arguments, using the
    // equivalence to a hypergeometric function.
    // See http://dlmf.nist.gov/8.17#ii
    template<typename T>
    void gradIncBeta(T& g1, T& g2, T a, T b, T z)
    {
      using std::log;
      using std::exp;

      T c1 = log(z);
      T c2 = log(1 - z);
      T c3 = boost::math::beta(a, b, z);
          
      T C = exp( a * c1 + b * c2 ) / a;
          
      T dF1 = 0;
      T dF2 = 0;
          
      if(C) grad2F1(dF1, dF2, a + b, 1, a + 1, z);

          
      g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
      g2 = c2 * c3 + C * dF1;
          
    }
      
    // Gradient of the regularized incomplete beta function ibeta(a, b, z)
    template<typename T>
    void gradRegIncBeta(T& g1, T& g2, T a, T b, T z, 
                        T digammaA, T digammaB, T digammaSum, T betaAB)
    {
          
      T dBda = 0;
      T dBdb = 0;
          
      gradIncBeta(dBda, dBdb, a, b, z);
          
      T b1 = boost::math::beta(a, b, z);
          
      g1 = ( dBda - b1 * (digammaA - digammaSum) ) / betaAB;
      g2 = ( dBdb - b1 * (digammaB - digammaSum) ) / betaAB;
          
    }
      
    // Gradient of the regularized incomplete gamma functions igamma(a, g)
    // Precomputed values
    // g   = boost::math::tgamma(a)
    // dig = boost::math::digamma(a)
    template<typename T>
    T gradRegIncGamma(T a, T z, T g, T dig, 
                           T precision = 1e-6) {
      using stan::math::gamma_p;
      using std::log;
      using std::exp;

      T S = 0;
      T s = 1;
      T l = log(z);
          
      int k = 0;
      T delta = s / (a * a);
          
      while (fabs(delta) > precision) {
        S += delta;
        ++k;
        s *= - z / k;
        delta = s / ((k + a) * (k + a));
        if (boost::math::isinf(delta))
          throw std::domain_error("stan::math::gradRegIncGamma not converging");
      }
      return gamma_p(a, z) * ( dig - l ) + exp( a * l ) * S / g;
    }
      
  }

}

#endif
