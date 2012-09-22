#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__BERNOULLI_HPP__

#include <stan/agrad.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>

namespace stan {

  namespace prob {

    // Bernoulli(n|theta)   [0 <= n <= 1;   0 <= theta <= 1]
    // FIXME: documentation
    template <bool Prop,
              typename T_n, typename T_prob, 
              class Policy>
    typename return_type<T_prob>::type
    bernoulli_log(const T_n& n,
                  const T_prob& theta, 
                  const Policy&) {
      static const char* function = "stan::prob::bernoulli_log(%1%)";

      using stan::math::check_finite;
      using stan::math::check_bounded;
      using stan::math::log1m;
      using stan::math::value_of;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      
      // check if any vectors are zero length
      if (!(stan::length(n)
	    && stan::length(theta)))
	return 0.0;
      
      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      if (!check_bounded(function, n, 0, 1, "Random variable", &logp, Policy()))
        return logp;
      if (!check_finite(function, theta, "Probability parameter", &logp, Policy()))
        return logp;
      if (!check_bounded(function, theta, 0.0, 1.0,
                         "Probability parameter", &logp, Policy()))
        return logp;
      if (!(check_consistent_sizes(function,
                                   n,theta,
				   "Random variable","Probability parameter",
                                   &logp, Policy())))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<Prop,T_prob>::value)
	return 0.0;

      // set up template expressions wrapping scalars into vector views
      VectorView<const T_n> n_vec(n);
      VectorView<const T_prob> theta_vec(theta);
      size_t N = max_size(n, theta);
      agrad::OperandsAndPartials1<T_prob>
        operands_and_partials(theta, theta_vec);

      for (size_t n = 0; n < N; n++) {
	// pull out values of arguments
	const int n_int = value_of(n_vec[n]);
	const double theta_dbl = value_of(theta_vec[n]);
	
	if (include_summand<Prop,T_prob>::value) {
	  if (n_int == 1)
	    logp += log(theta_dbl);
	  else
	    logp += log1m(theta_dbl);
	}

	// gradient
	if (!is_constant<typename is_vector<T_prob>::type>::value) {
	  if (n_int == 1)
	    operands_and_partials.d_x1[n] += 1.0 / theta_dbl;
	  else
	    operands_and_partials.d_x1[n] += 1.0 / (theta_dbl - 1.0);
	}
      }
      return operands_and_partials.to_var(logp);
    }


    template <bool Prop,
	      typename T_y,
              typename T_prob>
    inline
    typename return_type<T_prob>::type
    bernoulli_log(const T_y& n, 
                  const T_prob& theta) {
      return bernoulli_log<Prop>(n,theta,stan::math::default_policy());
    }


    template <typename T_y,
	      typename T_prob, 
              class Policy>
    inline
    typename return_type<T_prob>::type
    bernoulli_log(const T_y& n, 
                  const T_prob& theta, 
                  const Policy&) {
      return bernoulli_log<false>(n,theta,Policy());
    }


    template <typename T_y, typename T_prob>
    inline
    typename return_type<T_prob>::type
    bernoulli_log(const T_y& n, 
                  const T_prob& theta) {
      return bernoulli_log<false>(n,theta,stan::math::default_policy());
    }

    // Bernoulli(n|inv_logit(theta))   [0 <= n <= 1;   -inf <= theta <= inf]
    template <bool propto,
              typename T_prob, 
              class Policy>
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_logit_log(const int n, 
                        const T_prob& theta, 
                        const Policy&) {
      static const char* function = "stan::prob::bernoulli_logit_log(%1%)";

      using stan::math::check_not_nan;
      using stan::math::check_bounded;

      T_prob lp;
      if (!check_bounded(function, n, 0, 1, "n", &lp, Policy()))
        return lp;
      if (!check_not_nan(function, theta, "Logit transformed probability parameter",
                         &lp, Policy()))
        return lp;

      using stan::math::log1m;

      if (include_summand<propto,T_prob>::value) {
        T_prob ntheta = (2*n-1) * theta;
        // Handle extreme values gracefully using Taylor approximations.
        const static double cutoff = 20.0;
        if (ntheta > cutoff)
          return -exp(-ntheta);
        else if (ntheta < -cutoff)
          return ntheta;
        else
          return -log(1 + exp(-ntheta));
      }
      return 0.0;
    }


    template <bool propto,
              typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_logit_log(const int n, 
                        const T_prob& theta) {
      return bernoulli_logit_log<propto>(n,theta,stan::math::default_policy());
    }


    template <typename T_prob, 
              class Policy>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_logit_log(const int n, 
                        const T_prob& theta, 
                        const Policy&) {
      return bernoulli_logit_log<false>(n,theta,Policy());
    }


    template <typename T_prob>
    inline
    typename boost::math::tools::promote_args<T_prob>::type
    bernoulli_logit_log(const int n, 
                        const T_prob& theta) {
      return bernoulli_logit_log<false>(n,theta,stan::math::default_policy());
    }


  }
}
#endif
