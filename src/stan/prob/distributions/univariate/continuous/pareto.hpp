#ifndef __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__
#define __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__

#include <stan/prob/constants.hpp>
#include <stan/maths/error_handling.hpp>
#include <stan/prob/traits.hpp>

namespace stan {
  namespace prob {

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <bool propto = false,
              typename T_y, typename T_scale, typename T_shape, 
              class Policy = stan::maths::default_policy>
    inline typename boost::math::tools::promote_args<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, const Policy& = Policy()) {
      static const char* function = "stan::prob::pareto_log<%1%>(%1%)";
      
      using stan::maths::check_positive;
      using stan::maths::check_not_nan;
      using boost::math::tools::promote_args;
      
      typename promote_args<T_y,T_scale,T_shape>::type lp;
      if (!check_positive(function, y_min, "Scale parameter, y_min,", &lp, Policy()))
        return lp;
      if (!check_positive(function, alpha, "Shape parameter, alpha,", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, y, "Random variate, y,", &lp, Policy()))
        return lp;
      
      if (y < y_min)
        return LOG_ZERO;
          
      using stan::maths::multiply_log;

      lp = 0.0;
      if (include_summand<propto,T_shape>::value)
        lp += log(alpha);
      if (include_summand<propto,T_scale,T_shape>::value)
        lp += multiply_log(alpha, y_min);
      if (include_summand<propto,T_y,T_shape>::value)
        lp -= multiply_log(alpha+1.0, y);
      return lp;
    }

  }
}
#endif
