#ifndef STAN_MATH_PRIM_SCAL_FUN_INC_BETA_DERIVATIVES_HPP
#define STAN_MATH_PRIM_SCAL_FUN_INC_BETA_DERIVATIVES_HPP

#include <stan/math/prim/scal/fun/lgamma.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <cmath>

namespace stan {
  namespace math {
    

    template <typename T>
    T inc_beta_ddz(T a, T b, T z) {
      using std::exp;
      using std::log;
      return exp((b - 1) * log(1 - z) + (a - 1) * log(z)
                 + lgamma(a + b) - lgamma(a) - lgamma(b));
    }

    template <>
    double inc_beta_ddz(double a, double b, double z) {
      using boost::math::ibeta_derivative;
      return ibeta_derivative(a, b, z);
    }

  }  // math
}   // stan

#endif
