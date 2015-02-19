#ifndef STAN__MATH__PRIM__SCAL__PROB__HYPERGEOMETRIC_RNG_HPP
#define STAN__MATH__PRIM__SCAL__PROB__HYPERGEOMETRIC_RNG_HPP

#include <vector>
#include <boost/math/distributions.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_bounded.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/meta/length.hpp>
#include <stan/math/prim/scal/meta/is_constant_struct.hpp>
#include <stan/math/prim/scal/meta/VectorView.hpp>
#include <stan/math/prim/scal/meta/VectorBuilder.hpp>
#include <stan/math/prim/scal/meta/partials_return_type.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/uniform_rng.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline int
    hypergeometric_rng(const int N,
                       const int a,
                       const int b,
                       RNG& rng) {
      using boost::variate_generator;
      
      static const char* function("stan::prob::hypergeometric_rng");

      using stan::math::check_bounded;
      using stan::math::check_positive;

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
