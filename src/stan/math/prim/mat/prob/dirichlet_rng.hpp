#ifndef STAN_MATH_PRIM_MAT_PROB_DIRICHLET_RNG_HPP
#define STAN_MATH_PRIM_MAT_PROB_DIRICHLET_RNG_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <boost/random/gamma_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/mat/err/check_simplex.hpp>
#include <stan/math/prim/scal/err/check_consistent_sizes.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace prob {

    template <class RNG>
    inline Eigen::VectorXd
    dirichlet_rng(const Eigen::Matrix<double, Eigen::Dynamic, 1>& alpha,
                  RNG& rng) {
      using boost::variate_generator;
      using boost::gamma_distribution;

      double sum = 0;
      Eigen::VectorXd y(alpha.rows());
      for (int i = 0; i < alpha.rows(); i++) {
        variate_generator<RNG&, gamma_distribution<> >
          gamma_rng(rng, gamma_distribution<>(alpha(i, 0), 1));
        y(i) = gamma_rng();
        sum += y(i);
      }

      for (int i = 0; i < alpha.rows(); i++)
        y(i) /= sum;
      return y;
    }
  }
}
#endif
