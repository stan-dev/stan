#ifndef __STAN__PROB__INTERNAL_MATH__MATH__GRAD_REG_INC_GAMMA_HPP__
#define __STAN__PROB__INTERNAL_MATH__MATH__GRAD_REG_INC_GAMMA_HPP__

#include <math.h>
#include <stan/math/functions/gamma_p.hpp>

namespace stan {
    
  namespace math {

    // Gradient of the regularized incomplete gamma functions igamma(a, g)
    // Precomputed values
    // g   = boost::math::tgamma(a)
    // dig = boost::math::digamma(a)
    template<typename T>
    T grad_reg_inc_gamma(T a, T z, T g, T dig, 
                         T precision = 1e-6) {
      using stan::math::gamma_p;
      using std::log;
      using std::exp;
      using std::fabs;

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
