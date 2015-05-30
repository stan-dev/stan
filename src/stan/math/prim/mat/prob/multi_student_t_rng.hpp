#ifndef STAN_MATH_PRIM_MAT_PROB_MULTI_STUDENT_T_RNG_HPP
#define STAN_MATH_PRIM_MAT_PROB_MULTI_STUDENT_T_RNG_HPP

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/mat/err/check_ldlt_factor.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/mat/fun/multiply.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/mat/fun/subtract.hpp>
#include <stan/math/prim/scal/fun/log1p.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/mat/prob/multi_normal_rng.hpp>
#include <stan/math/prim/scal/prob/inv_gamma_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <cstdlib>

namespace stan {

  namespace math {
    using Eigen::Dynamic;

    template <class RNG>
    inline Eigen::VectorXd
    multi_student_t_rng(const double nu,
                        const Eigen::Matrix<double, Dynamic, 1>& mu,
                        const Eigen::Matrix<double, Dynamic, Dynamic>& s,
                        RNG& rng) {
      static const char* function("stan::math::multi_student_t_rng");

      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_positive;

      check_finite(function, "Location parameter", mu);
      check_symmetric(function, "Scale parameter", s);
      check_not_nan(function, "Degrees of freedom parameter", nu);
      check_positive(function, "Degrees of freedom parameter", nu);

      Eigen::VectorXd z(s.cols());
      z.setZero();

      double w = stan::math::inv_gamma_rng(nu / 2, nu / 2, rng);
      return mu + std::sqrt(w) * stan::math::multi_normal_rng(z, s, rng);
    }
  }
}
#endif
