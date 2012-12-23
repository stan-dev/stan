#ifndef __STAN__PROB__INTERNAL_MATH_HPP__
#define __STAN__PROB__INTERNAL_MATH_HPP__

#include <math.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>

namespace stan {
    
  namespace math {

      // Gradient of the hypergeometric function 2F1(a, b | c | z) with respect to a and c
      void grad2F1(double& gradA, double& gradC, double a, double b, double c, double z, double precision = 1e-6)
      {
          
          gradA = 0;
          gradC = 0;
          
          double gradAold = 0;
          double gradCold = 0;
          
          int k = 0;
          double tOld = 1.0;
          double tNew = (a * b) / c;
          
          while(fabs(tNew) > precision)
          {
              
              gradAold = tNew * (gradAold / tOld + 1.0 / (a + k) );
              gradCold = tNew * (gradCold / tOld - 1.0 / (c + k) );              
              
              gradA += gradAold;
              gradC += gradCold;
              
              ++k;
              
              tNew = ( (a + k) / (c + k) ) * ( (b + k) / (k + 1) ) * z * tOld;
              
              if(tNew == 0) break;
              
          }
          
      }
      
      // Gradient of the incomplete beta function beta(a, b, z)
      // with respect to the first two arguments, using the
      // equivalence to a hypergeometric function.
      // See http://dlmf.nist.gov/8.17#ii
      void gradIncBeta(double& g1, double& g2, double a, double b, double z)
      {
          
          double c1 = std::log(z);
          double c2 = std::log(1 - z);
          double c3 = boost::math::beta(a, b, z);
          
          double C = std::exp( a * c1 + b * c2 ) / a;
          
          double dF1 = 0;
          double dF2 = 0;
          
          grad2F1(dF1, dF2, a + b, 1, a + 1, z);
          
          g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
          g2 = c2 * c3 + C * dF1;
          
      }
      
      // Gradient of the regularized incomplete beta function ibeta(a, b, z)
      void gradRegIncBeta(double& g1, double& g2, double a, double b, double z, 
                          double digammaA, double digammaB, double digammaSum, double betaAB)
      {
          
          double dBda = 0;
          double dBdb = 0;
          
          gradIncBeta(dBda, dBdb, a, b, z);
          
          double b1 = boost::math::beta(a, b, z);
          
          g1 = ( dBda - b1 * (digammaA - digammaSum) ) / betaAB;
          g2 = ( dBdb - b1 * (digammaB - digammaSum) ) / betaAB;
          
      }
      
      // Gradient of the regularized incomplete gamma functions igamma(a, g)
      double gradRegIncGamma(double a, double z, double g, double dig, double precision = 1e-6) 
      {
          
          using boost::math::gamma_p;
          
          double S = 0;
          double s = 1;
          double l = std::log(z);
          
          int k = 0;
          double delta = s / (a * a);
          
          while (fabs(delta) > precision)
          {
              S += delta;
              ++k;
              s *= - z / k;
              delta = s / ((k + a) * (k + a));
          }
          
          // Precomputed values
          // dig -> digamma(a)
          // g   -> g(a)
          return gamma_p(a, z) * ( dig - l ) + std::exp( a * l ) * S / g;
          
      }
      
  }

}

#endif