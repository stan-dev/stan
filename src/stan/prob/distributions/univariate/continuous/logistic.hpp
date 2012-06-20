#ifndef __STAN__PROB__DISTRIBUTIONS__LOGISTIC_HPP__
#define __STAN__PROB__DISTRIBUTIONS__LOGISTIC_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // Logistic(y|mu,sigma)    [sigma > 0]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                 const Policy&) {
      static const char* function = "stan::prob::logistic_log<%1%>(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if (!check_finite(function, y, "Random variate y", &lp, Policy()))
        return lp;
      if (!check_finite(function, mu, "Location parameter, mu,",
                        &lp, Policy()))
        return lp;
      if (!check_finite(function, sigma, "Scale parameter, sigma,", &lp, 
                        Policy()))
        return lp;
      if (!check_positive(function, sigma, "Scale parameter, sigma,",
                          &lp, Policy()))
        return lp;

      using stan::math::log1p;

      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= (y - mu)/sigma;
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= 2.0 * log1p(exp(-(y - mu)/sigma));
      return lp;
    }

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log<propto>(y,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_loc, typename T_scale,
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
                 const Policy&) {
      return logistic_log<false>(y,mu,sigma,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    logistic_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return logistic_log<false>(y,mu,sigma,stan::math::default_policy());
    }

  }
}
#endif
