#ifndef STAN_MATH_PRIM_SCAL_FUN_GRAD_REG_INC_BETA_HPP
#define STAN_MATH_PRIM_SCAL_FUN_GRAD_REG_INC_BETA_HPP

#include <stan/math/prim/scal/fun/grad_inc_beta.hpp>
#include <stan/math/prim/scal/fun/inc_beta.hpp>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <cmath>

namespace stan {
  namespace math {

    // Gradient of the regularized incomplete beta function ibeta(a, b, z)
    template<typename T>
    void grad_reg_inc_beta(T& g1, T& g2, T a, T b, T z,
                           T digammaA, T digammaB, T digammaSum, T betaAB) {
      using stan::math::inc_beta;
      using stan::math::grad_inc_beta;
      using std::exp;
      using stan::math::lbeta;

      T dBda = 0;
      T dBdb = 0;
      grad_inc_beta(dBda, dBdb, a, b, z);
      T b1 = exp(lbeta(a, b)) * inc_beta(a, b, z);
      g1 = (dBda - b1 * (digammaA - digammaSum)) / betaAB;
      g2 = (dBdb - b1 * (digammaB - digammaSum)) / betaAB;
    }

  }
}
#endif
