#ifndef STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP
#define STAN__PROB__DISTRIBUTIONS__UNIVARIATE__DISCRETE__HYPERGEOMETRIC_HPP

#include <vector>
#include <boost/math/distributions.hpp>
#include <stan/error_handling/scalar/check_consistent_sizes.hpp>
#include <stan/error_handling/scalar/check_bounded.hpp>
#include <stan/error_handling/scalar/check_finite.hpp>
#include <stan/error_handling/scalar/check_greater.hpp>
#include <stan/error_handling/scalar/check_positive.hpp>
#include <stan/math/functions/binomial_coefficient_log.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>

namespace stan {

  namespace prob {

    // Hypergeometric(n|N,a,b)  [0 <= n <= a;  0 <= N-n <= b;  0 <= N <= a+b]
    // n: #white balls drawn;  N: #balls drawn;  
    // a: #white balls;  b: #black balls
    template <bool propto,
              typename T_n, typename T_N,
              typename T_a, typename T_b>
    double
    hypergeometric_log(const T_n& n, const T_N& N, 
                       const T_a& a, const T_b& b) {
      static const std::string function("stan::prob::hypergeometric_log");

      using stan::error_handling::check_finite;      
      using stan::error_handling::check_bounded;
      using stan::error_handling::check_greater;
      using stan::error_handling::check_consistent_sizes;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(n)
            && stan::length(N)
            && stan::length(a)
            && stan::length(b)))
        return 0.0;


      VectorView<const T_n> n_vec(n);
      VectorView<const T_N> N_vec(N);
      VectorView<const T_a> a_vec(a);
      VectorView<const T_b> b_vec(b);
      size_t size = max_size(n, N, a, b);
      
      double logp(0.0);
      check_bounded(function, "Successes variable", n, 0, a);
      check_greater(function, "Draws parameter", N, n);
      for (size_t i = 0; i < size; i++) {
        check_bounded(function, "Draws parameter minus successes variable", N_vec[i]-n_vec[i], 0, b_vec[i]);
        check_bounded(function, "Draws parameter", N_vec[i], 0, a_vec[i]+b_vec[i]);
      }
      check_consistent_sizes(function,
                             "Successes variable", n,
                             "Draws parameter", N,
                             "Successes in population parameter", a,
                             "Failures in population parameter", b);
      
      // check if no variables are involved and prop-to
      if (!include_summand<propto>::value)
        return 0.0;


      for (size_t i = 0; i < size; i++)
        logp += math::binomial_coefficient_log(a_vec[i],n_vec[i])
          + math::binomial_coefficient_log(b_vec[i],N_vec[i]-n_vec[i])
          - math::binomial_coefficient_log(a_vec[i]+b_vec[i],N_vec[i]);
      return logp;
    }

    template <typename T_n,
              typename T_N,
              typename T_a,
              typename T_b>
    inline
    double
    hypergeometric_log(const T_n& n, 
                       const T_N& N, 
                       const T_a& a, 
                       const T_b& b) {
      return hypergeometric_log<false>(n,N,a,b);
    }

    template <class RNG>
    inline int
    hypergeometric_rng(const int N,
                       const int a,
                       const int b,
                       RNG& rng) {
      using boost::variate_generator;
      
      static const std::string function("stan::prob::hypergeometric_rng");

      using stan::error_handling::check_bounded;
      using stan::error_handling::check_positive;

      check_bounded(function, "Draws parameter", N, 0, a+b);
      check_positive(function, "Draws parameter", N);
      check_positive(function, "Successes in population parameter", a);
      check_positive(function, "Failures in population parameter", b);

      boost::math::hypergeometric_distribution<>dist (b, N, a + b);
      std::vector<double> index(a);
      for(int i = 0; i < a; i++)
        index[i] = cdf(dist, i + 1);

      double c = uniform_rng(0.0, 1.0, rng);
      int min = 0;
      int max = a - 1;
      int mid = 0;
      while (min < max) {
        mid = (min + max) / 2;
        if(index[mid] > c)
          max = mid;
        else
          min = mid + 1;
      }
      return min + 1;
    }
  }
}
#endif
