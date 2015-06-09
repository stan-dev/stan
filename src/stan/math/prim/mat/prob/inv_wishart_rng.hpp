#ifndef STAN_MATH_PRIM_MAT_PROB_INV_WISHART_RNG_HPP
#define STAN_MATH_PRIM_MAT_PROB_INV_WISHART_RNG_HPP

#include <stan/math/prim/mat/err/check_ldlt_factor.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/mat/fun/log_determinant_ldlt.hpp>
#include <stan/math/prim/mat/fun/mdivide_left_ldlt.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/mat/prob/wishart_rng.hpp>

namespace stan {
  namespace math {

    template <class RNG>
    inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
    inv_wishart_rng(const double nu,
                    const Eigen::Matrix
                    <double, Eigen::Dynamic, Eigen::Dynamic>& S,
                    RNG& rng) {
      static const char* function("stan::math::inv_wishart_rng");

      using stan::math::check_greater;
      using stan::math::check_square;
      using Eigen::MatrixXd;
      using stan::math::index_type;

      typename index_type<MatrixXd>::type k = S.rows();

      check_greater(function, "Degrees of freedom parameter", nu, k-1);
      check_square(function, "scale parameter", S);

      MatrixXd S_inv = MatrixXd::Identity(k, k);
      S_inv = S.ldlt().solve(S_inv);

      return wishart_rng(nu, S_inv, rng).inverse();
    }
  }
}
#endif
