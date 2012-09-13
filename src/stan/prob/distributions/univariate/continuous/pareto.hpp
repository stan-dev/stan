#ifndef __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__
#define __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__

#include <stan/prob/constants.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_scale, typename T_shape, 
              class Policy>
    typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, 
               const Policy&) {
      static const char* function = "stan::prob::pareto_log<%1%>(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_scale,T_shape>::type lp;
      if (!check_not_nan(function, y, "Random variable", &lp, Policy()))
        return lp;
      if (!check_finite(function, y_min, "Scale parameter",
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, y_min, "Scale parameter", 
                          &lp, Policy()))
        return lp;
      if (!check_finite(function, alpha, "Shape parameter", 
                        &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, "Shape parameter", 
                          &lp, Policy()))
        return lp;
      
      if (y < y_min)
        return LOG_ZERO;
          
      using stan::math::multiply_log;

      lp = 0.0;
      if (include_summand<propto,T_shape>::value)
        lp += log(alpha);
      if (include_summand<propto,T_scale,T_shape>::value)
        lp += multiply_log(alpha, y_min);
      if (include_summand<propto,T_y,T_shape>::value)
        lp -= multiply_log(alpha+1.0, y);
      return lp;
    }


    template <bool propto,
              typename T_y, typename T_scale, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      return pareto_log<propto>(y,y_min,alpha,stan::math::default_policy());
    }

    template <typename T_y, typename T_scale, typename T_shape, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, 
               const Policy&) {
      return pareto_log<false>(y,y_min,alpha,Policy());
    }

    template <typename T_y, typename T_scale, typename T_shape>
    inline
    typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      return pareto_log<false>(y,y_min,alpha,stan::math::default_policy());
    }


  }
}
#endif
