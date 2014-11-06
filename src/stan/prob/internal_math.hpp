#ifndef STAN__PROB__INTERNAL_MATH_HPP
#define STAN__PROB__INTERNAL_MATH_HPP

#include <math.h>
#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/fpclassify.hpp>

namespace stan {
    
  namespace math {

      
      double F32(double a, double b, double c, double d, double e, double z, double precision = 1e-6)
      {
          
          double F = 1;
          
          double tNew = 0;
          
          double logT = 0;
          
          double logZ = std::log(z);
          
          int k = 0;
          
          while ( (fabs(tNew) > precision) || (k == 0) )
          {
              
              double p = (a + k) * (b + k) * (c + k) / ( (d + k) * (e + k) * (k + 1) );
              
              // If a, b, or c is a negative integer then the series terminates
              // after a finite number of interations
              if(p == 0) break;
              
              logT += (p > 0 ? 1 : -1) * std::log(fabs(p)) + logZ;
              
              tNew = std::exp(logT);
              
              F += tNew;
              
              ++k;

          }
          
          return F;
          
      }
      
      void gradF32(double* g, double a, double b, double c, double d, double e, double z, double precision = 1e-6)
      {
          
          double gOld[6];
          
          for(double *p = g; p != g + 6; ++p) *p = 0;
          for(double *p = gOld; p != gOld + 6; ++p) *p = 0;
          
          double tOld = 1;
          double tNew = 0;
          
          double logT = 0;
          
          double logZ = std::log(z);
          
          int k = 0;
          
          while ( (fabs(tNew) > precision) || (k == 0) )
          {
              
              double C = (a + k) / (d + k);
              C *= (b + k) / (e + k);
              C *= (c + k) / (1 + k);
              
              // If a, b, or c is a negative integer then the series terminates
              // after a finite number of interations
              if(C == 0) break;
              
              logT += (C > 0 ? 1 : -1) * std::log(fabs(C)) + logZ;
              
              tNew = std::exp(logT);
              
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
      void grad2F1(double& gradA, double& gradC, double a, double b, double c, double z, double precision = 1e-6)
      {
          
          gradA = 0;
          gradC = 0;
          
          double gradAold = 0;
          double gradCold = 0;
          
          int k = 0;
          double tDak = 1.0 / (a - 1);
          
          while ( (fabs(tDak * (a + (k - 1)) ) > precision) || (k == 0) )
          {
              
              const double r = ( (a + k) / (c + k) ) * ( (b + k) / (double)(k + 1) ) * z;
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
      void gradIncBeta(double& g1, double& g2, double a, double b, double z)
      {
          
          double c1 = std::log(z);
          double c2 = std::log(1 - z);
          double c3 = boost::math::beta(a, b, z);
          
          double C = std::exp( a * c1 + b * c2 ) / a;
          
          double dF1 = 0;
          double dF2 = 0;
          
          if(C) grad2F1(dF1, dF2, a + b, 1, a + 1, z);

          
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
    // Precomputed values
    // g   = boost::math::tgamma(a)
    // dig = boost::math::digamma(a)
    double gradRegIncGamma(double a, double z, double g, double dig, 
                           double precision = 1e-6) {
      using boost::math::gamma_p;
          
      double S = 0;
      double s = 1;
      double l = std::log(z);
          
      int k = 0;
      double delta = s / (a * a);
          
      while (fabs(delta) > precision) {
        S += delta;
        ++k;
        s *= - z / k;
        delta = s / ((k + a) * (k + a));
        if (boost::math::isinf(delta))
          throw std::domain_error("stan::math::gradRegIncGamma not converging");
      }
      return gamma_p(a, z) * ( dig - l ) + std::exp( a * l ) * S / g;
    }
      
  }

}

#endif
