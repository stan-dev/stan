#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__MULTINOMIAL_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__MULTINOMIAL_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/math/error_handling.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/matrix_error_handling.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>
#include <stan/prob/distributions/multivariate/discrete/categorical.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {
    // Multinomial(ns|N,theta)   [0 <= n <= N;  SUM ns = N;   
    //                            0 <= theta[n] <= 1;  SUM theta = 1]
    template <bool propto,
              typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    multinomial_log(const std::vector<int>& ns,
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      static const char* function = "stan::prob::multinomial_log(%1%)";

      using stan::math::check_nonnegative;
      using stan::math::check_simplex;
      using stan::math::check_size_match;
      using boost::math::tools::promote_args;
      using boost::math::lgamma;

      typename promote_args<T_prob>::type lp(0.0);
      check_nonnegative(function, ns, "Number of trials variable", &lp);
      check_simplex(function, theta, "Probabilites parameter", 
                    &lp);
      check_size_match(function, 
                       ns.size(), "Size of number of trials variable",
                       theta.rows(), "rows of probabilities parameter",
                       &lp);
      using stan::math::multiply_log;

      if (include_summand<propto>::value) {     
        double sum = 1.0;
        for (unsigned int i = 0; i < ns.size(); ++i) 
          sum += ns[i];
        lp += lgamma(sum);
        for (unsigned int i = 0; i < ns.size(); ++i)
          lp -= lgamma(ns[i] + 1.0);
      }
      if (include_summand<propto,T_prob>::value)
        for (unsigned int i = 0; i < ns.size(); ++i)
          lp += multiply_log(ns[i], theta[i]);
      return lp;
    }

    template <typename T_prob>
    typename boost::math::tools::promote_args<T_prob>::type
    multinomial_log(const std::vector<int>& ns,
                    const Eigen::Matrix<T_prob,Eigen::Dynamic,1>& theta) {
      return multinomial_log<false>(ns,theta);
    }

    template <class RNG>
    inline std::vector<int>
    multinomial_rng(const Eigen::Matrix<double,Eigen::Dynamic,1>& theta,
                    const int N,
                    RNG& rng) {
      static const char* function = "stan::prob::multinomial_rng(%1%)";
      using stan::math::check_simplex;
      using stan::math::check_positive;

      check_simplex(function, theta, "Probabilites parameter", (double*)0);
      check_positive(function,N,"number of trials variables", (double*)0);

      std::vector<int> result(theta.size(),0);
      double mass_left = 1.0;
      int n_left = N;
      for (int k = 0; n_left > 0 && k < theta.size(); ++k) {
        double p = theta[k] / mass_left;
        if (p > 1.0) p = 1.0;
        result[k] = binomial_rng(n_left,p,rng);
        n_left -= result[k];
        mass_left -= theta[k];
      }
      return result;
    }


  }
}
#endif
