#ifndef STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__MULTINOMIAL_HPP
#define STAN__PROB__DISTRIBUTIONS__MULTIVARIATE__DISCRETE__MULTINOMIAL_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/error_handling/matrix/check_simplex.hpp>
#include <stan/error_handling/matrix/check_size_match.hpp>
#include <stan/error_handling/scalar/check_nonnegative.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/error_handling/matrix.hpp>
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
      static const std::string function("stan::prob::multinomial_log");

      using stan::error_handling::check_nonnegative;
      using stan::error_handling::check_simplex;
      using stan::error_handling::check_size_match;
      using boost::math::tools::promote_args;
      using boost::math::lgamma;

      typename promote_args<T_prob>::type lp(0.0);
      check_nonnegative(function, "Number of trials variable", ns);
      check_simplex(function, "Probabilites parameter", theta);
      check_size_match(function, 
                       "Size of number of trials variable", ns.size(),
                       "rows of probabilities parameter", theta.rows());
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
      static const std::string function("stan::prob::multinomial_rng");
      using stan::error_handling::check_simplex;
      using stan::error_handling::check_positive;

      check_simplex(function, "Probabilites parameter", theta);
      check_positive(function, "number of trials variables", N);

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
