#ifndef STAN_MATH_PRIM_MAT_PROB_MULTI_NORMAL_CHOLESKY_RNG_HPP
#define STAN_MATH_PRIM_MAT_PROB_MULTI_NORMAL_CHOLESKY_RNG_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/mat/fun/columns_dot_product.hpp>
#include <stan/math/prim/mat/fun/columns_dot_self.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/mat/fun/dot_self.hpp>
#include <stan/math/prim/mat/fun/log.hpp>
#include <stan/math/prim/mat/fun/log_determinant.hpp>
#include <stan/math/prim/mat/fun/mdivide_left_spd.hpp>
#include <stan/math/prim/mat/fun/mdivide_left_tri_low.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/subtract.hpp>
#include <stan/math/prim/mat/fun/sum.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {
    using Eigen::Dynamic;

    template <class RNG>
    inline Eigen::VectorXd
    multi_normal_cholesky_rng(const Eigen::Matrix<double, Dynamic, 1>& mu,
                              const Eigen::Matrix<double, Dynamic, Dynamic>& S,
                              RNG& rng) {
      using boost::variate_generator;
      using boost::normal_distribution;

      static const char* function("stan::math::multi_normal_cholesky_rng");

      using stan::math::check_finite;

      check_finite(function, "Location parameter", mu);

      variate_generator<RNG&, normal_distribution<> >
        std_normal_rng(rng, normal_distribution<>(0, 1));

      Eigen::VectorXd z(S.cols());
      for (int i = 0; i < S.cols(); i++)
        z(i) = std_normal_rng();

      return mu + S * z;
    }
  }
}

#endif
