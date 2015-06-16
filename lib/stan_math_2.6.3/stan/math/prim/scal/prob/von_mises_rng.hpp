#ifndef STAN_MATH_PRIM_SCAL_PROB_VON_MISES_RNG_HPP
#define STAN_MATH_PRIM_SCAL_PROB_VON_MISES_RNG_HPP

#include <stan/math/prim/scal/meta/OperandsAndPartials.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_nonnegative.hpp>
#include <stan/math/prim/scal/err/check_positive_finite.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/fun/modified_bessel_first_kind.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cmath>

namespace stan {

  namespace math {

    // The algorithm used in von_mises_rng is a modified version of the
    // algorithm in:
    //
    // Efficient Simulation of the von Mises Distribution
    // D. J. Best and N. I. Fisher
    // Journal of the Royal Statistical Society. Series C (Applied Statistics),
    // Vol. 28, No. 2 (1979), pp. 152-157
    //
    // See licenses/stan-license.txt for Stan license.

    template <class RNG>
    inline double
    von_mises_rng(const double mu,
                  const double kappa,
                  RNG& rng) {
      using boost::variate_generator;
      using stan::math::uniform_rng;
      using std::fmod;
      using std::log;
      using std::pow;

      static const char* function("stan::math::von_mises_rng");

      stan::math::check_finite(function, "mean", mu);
      stan::math::check_positive_finite(function, "inverse of variance", kappa);

      double r = 1 + pow((1 + 4 * kappa * kappa), 0.5);
      double rho = 0.5 * (r - pow(2 * r, 0.5)) / kappa;
      double s = 0.5 * (1 + rho * rho) / rho;

      bool done = 0;
      double W;
      while (!done) {
        double Z = std::cos(stan::math::pi() * uniform_rng(0.0, 1.0, rng));
        W = (1 + s * Z) / (s + Z);
        double Y = kappa * (s - W);
        double U2 = uniform_rng(0.0, 1.0, rng);
        done = Y * (2 - Y) - U2 > 0;

        if (!done)
          done = log(Y / U2) + 1 - Y >= 0;
      }

      double U3 = uniform_rng(0.0, 1.0, rng) - 0.5;
      double sign = ((U3 >= 0) - (U3 <= 0));

      //  it's really an fmod() with a positivity constraint
      return sign * std::acos(W)
        + fmod(fmod(mu, 2*stan::math::pi())+2*stan::math::pi(),
               2*stan::math::pi());
    }

  }
}
#endif
