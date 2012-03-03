#ifndef __STAN__PROB__DISTRIBUTIONS__CAUCHY_HPP__
#define __STAN__PROB__DISTRIBUTIONS__CAUCHY_HPP__

#include <stan/prob/traits.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Cauchy(y|mu,sigma)  [sigma > 0]
    template <bool propto,
              typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma, 
               const Policy&) {
      static const char* function = "stan::prob::cauchy_log<%1%>(%1%)";
      
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if(!check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;
      if(!check_finite(function, mu, "Location parameter, mu,", 
                       &lp, Policy()))
        return lp;
      if(!check_finite(function, sigma, "Scale parameter, sigma,", 
                       &lp, Policy()))
        return lp;
      if(!check_positive(function, sigma, "Scale parameter, sigma,", 
                         &lp, Policy()))
        return lp;

      using stan::math::log1p;
      using stan::math::square;
      
      if (include_summand<propto>::value)
        lp += NEG_LOG_PI;
      if (include_summand<propto,T_scale>::value)
        lp -= log(sigma);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
        lp -= log1p(square((y - mu) / sigma));
      return lp;
    }


    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_log<propto>(y,mu,sigma,stan::math::default_policy());
    }

    template <typename T_y, typename T_loc, typename T_scale, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
               const Policy&) {
      return cauchy_log<false>(y,mu,sigma,Policy());
    }

    template <typename T_y, typename T_loc, typename T_scale>
    inline
    typename boost::math::tools::promote_args<T_y,T_loc,T_scale>::type
    cauchy_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      return cauchy_log<false>(y,mu,sigma,stan::math::default_policy());
    }


  }
}
#endif
