#ifndef __STAN__PROB__DISTRIBUTIONS__BERNOULLI_HPP__
#define __STAN__PROB__DISTRIBUTIONS__BERNOULLI_HPP__

#include <stan/maths/special_functions.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/error_handling.hpp>
#include <stan/prob/constants.hpp>

namespace stan {
  namespace prob {
    using boost::math::tools::promote_args;
    using boost::math::policies::policy;

    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    template <bool propto = false,
              typename T_prob, 
              class Policy = policy<> > 
    inline typename boost::math::tools::promote_args<T_prob>::type
      bernoulli_log(const unsigned int n, const T_prob& theta, const Policy& = Policy()) {
      using stan::maths::log1m;
      // FIXME: domain checks           
      if (include_summand<propto,T_prob>::value) {
        if (n == 1)
          return log(theta);
        else if (n == 0)
          return log1m(theta);
      }
      return 0.0;
    }


  }
}
#endif
