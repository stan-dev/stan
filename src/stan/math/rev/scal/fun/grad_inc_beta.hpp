#ifndef STAN_MATH_REV_SCAL_FUN_GRAD_INC_BETA_HPP
#define STAN_MATH_REV_SCAL_FUN_GRAD_INC_BETA_HPP

#include <math.h>
#include <stan/math/prim/scal/fun/lbeta.hpp>
#include <stan/math/rev/scal/fun/exp.hpp>
#include <stan/math/rev/scal/fun/fabs.hpp>
#include <stan/math/rev/scal/fun/lgamma.hpp>
#include <stan/math/rev/scal/fun/log.hpp>
#include <stan/math/rev/scal/fun/log1m.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/scal/fun/value_of.hpp>
#include <stan/math/prim/scal/fun/value_of.hpp>

#include <stan/math/rev/scal/fun/inc_beta.hpp>
#include <stan/math/prim/scal/fun/grad_2F1.hpp>

namespace stan {

  namespace math {

    // Gradient of the incomplete beta function beta(a, b, z)
    // with respect to the first two arguments, using the
    // equivalence to a hypergeometric function.
    // See http://dlmf.nist.gov/8.17#ii
    void grad_inc_beta(stan::math::var& g1,
                       stan::math::var& g2,
                       stan::math::var a,
                       stan::math::var b,
                       stan::math::var z) {
      using stan::math::value_of;
      using stan::math::value_of;

      stan::math::var c1 = stan::math::log(z);
      stan::math::var c2 = stan::math::log1m(z);
      stan::math::var c3 = stan::math::exp(stan::math::lbeta(a, b))
        * stan::math::inc_beta(a, b, z);

      stan::math::var C = exp(a * c1 + b * c2) / a;

      stan::math::var dF1 = 0;
      stan::math::var dF2 = 0;

      if (value_of(value_of(C))) stan::math::grad_2F1(dF1, dF2, a + b,
                                                    (stan::math::var)1.0,
                                                    a + 1, z);


      g1 = (c1 - 1.0 / a) * c3 + C * (dF1 + dF2);
      g2 = c2 * c3 + C * dF1;
    }

  }

}

#endif
