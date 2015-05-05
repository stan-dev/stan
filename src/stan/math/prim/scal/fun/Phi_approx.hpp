#ifndef STAN_MATH_PRIM_SCAL_FUN_PHI_APPROX_HPP
#define STAN_MATH_PRIM_SCAL_FUN_PHI_APPROX_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>

namespace stan {
  namespace math {

    /**
     * Approximation of the unit normal CDF.
     *
     * http://www.jiem.org/index.php/jiem/article/download/60/27
     *
     * This function can be used to implement the inverse link function
     * for probit regression.
     *
     * @param x Argument.
     * @return Probability random sample is less than or equal to argument.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    Phi_approx(T x) {
      using std::pow;
      return inv_logit(0.07056 * pow(x, 3.0) + 1.5976 * x);
    }

  }
}

#endif
