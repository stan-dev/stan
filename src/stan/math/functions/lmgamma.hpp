#ifndef STAN__MATH__FUNCTIONS__LMGAMMA_HPP
#define STAN__MATH__FUNCTIONS__LMGAMMA_HPP

#include <boost/math/tools/promotion.hpp>
#include <stan/math/constants.hpp>
#include <boost/math/special_functions/gamma.hpp>

namespace stan {

  namespace math {

    /**
     * Return the natural logarithm of the multivariate gamma function
     * with the speciifed dimensions and argument.
     *
     * <p>The multivariate gamma function \f$\Gamma_k(x)\f$ for
     * dimensionality \f$k\f$ and argument \f$x\f$ is defined by
     *
     * <p>\f$\Gamma_k(x) = \pi^{k(k-1)/4} \, \prod_{j=1}^k \Gamma(x + (1 - j)/2)\f$,
     *
     * where \f$\Gamma()\f$ is the gamma function.
     *
     * @param k Number of dimensions.
     * @param x Function argument.
     * @return Natural log of the multivariate gamma function.
     * @tparam T Type of scalar.
     */
    template <typename T>
    inline typename boost::math::tools::promote_args<T>::type
    lmgamma(const int k, T x) {
      using boost::math::lgamma;
      typename boost::math::tools::promote_args<T>::type result 
        = k * (k - 1) * LOG_PI_OVER_FOUR;

      for (int j = 1; j <= k; ++j)
        result += lgamma(x + (1.0 - j) / 2.0);
      return result;
    }
      
  }
}
#endif
