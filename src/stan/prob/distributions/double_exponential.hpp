#ifndef __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__
#define __STAN__PROB__DISTRIBUTIONS__DOUBLE_EXPONENTIAL_HPP__

#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>



namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // DoubleExponential(y|mu,sigma)  [sigma > 0]
    template <bool propto = false,
	      typename T_y, typename T_loc, typename T_scale, 
	      class Policy = policy<> > 
    inline typename promote_args<T_y,T_loc,T_scale>::type
    double_exponential_log(const T_y& y, const T_loc& mu, const T_scale& sigma, const Policy& = Policy()) {
      //static const char* function = "stan::prob::double_exponential_log<%1%>(%1%)";
      
      //using stan::maths::abs;
      // FIXME: domain checks
      typename promote_args<T_y,T_loc,T_scale>::type lp(0.0);
      if (include_summand<propto>::value)
	lp += NEG_LOG_TWO;
      if (include_summand<propto,T_scale>::value)
	lp -= log(sigma);
      if (include_summand<propto,T_y,T_loc,T_scale>::value)
	lp -= fabs(y - mu) / sigma;
      return lp;
    }
    
    

  }
}
#endif
