#ifndef __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__
#define __STAN__PROB__DISTRIBUTIONS__PARETO_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>


namespace stan {
  namespace prob {

    // Pareto(y|y_m,alpha)  [y > y_m;  y_m > 0;  alpha > 0]
    template <bool propto,
              typename T_y, typename T_scale, typename T_shape, 
              class Policy>
    typename return_type<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha, 
               const Policy&) {
      static const char* function = "stan::prob::pareto_log(%1%)";
      
      using stan::math::check_finite;
      using stan::math::check_positive;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      
      typename return_type<T_y,T_scale,T_shape>::type logp(0.0);
      if (!check_not_nan(function, y, "Random variable", &logp, Policy()))
        return logp;
      if (!check_finite(function, y_min, "Scale parameter",
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, y_min, "Scale parameter", 
                          &logp, Policy()))
        return logp;
      if (!check_finite(function, alpha, "Shape parameter", 
                        &logp, Policy()))
        return logp;
      if (!check_positive(function, alpha, "Shape parameter", 
                          &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
				   y,y_min,alpha,
				   "Random variable","Scale parameter","Shape parameter",
                                   &logp, Policy())))
        return logp;

          
      using stan::math::multiply_log;

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> y_min_vec(y_min);
      VectorView<const T_shape> alpha_vec(alpha);
      size_t N = max_size(y, y_min, alpha);

      for (size_t n = 0; n < N; n++) {
	if (include_summand<propto,T_y,T_scale>::value)
	  if (y_vec[n] < y_min_vec[n])
	    return LOG_ZERO;
      
	if (include_summand<propto,T_shape>::value)
	  logp += log(alpha_vec[n]);
	if (include_summand<propto,T_scale,T_shape>::value)
	  logp += multiply_log(alpha_vec[n], y_min_vec[n]);
	if (include_summand<propto,T_y,T_shape>::value)
	  logp -= multiply_log(alpha_vec[n]+1.0, y_vec[n]);
      }
      return logp;
    }


    template <bool propto,
              typename T_y, typename T_scale, typename T_shape>
    inline
    typename return_type<T_y,T_scale,T_shape>::type
    pareto_log(const T_y& y, const T_scale& y_min, const T_shape& alpha) {
      return pareto_log<propto>(y,y_min,alpha,stan::math::default_policy());
    }

    template <typename T_y, typename T_scale, typename T_shape, 
              class Policy>
    inline
    typename return_type<T_y,T_scale,T_shape>::type
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
