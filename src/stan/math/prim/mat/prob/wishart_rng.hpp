#ifndef STAN_MATH_PRIM_MAT_PROB_WISHART_RNG_HPP
#define STAN_MATH_PRIM_MAT_PROB_WISHART_RNG_HPP

#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/mat/err/check_ldlt_factor.hpp>
#include <stan/math/prim/mat/err/check_square.hpp>
#include <stan/math/prim/scal/err/check_greater.hpp>
#include <stan/math/prim/scal/fun/lmgamma.hpp>
#include <stan/math/prim/mat/fun/crossprod.hpp>
#include <stan/math/prim/mat/fun/columns_dot_product.hpp>
#include <stan/math/prim/mat/fun/trace.hpp>
#include <stan/math/prim/mat/fun/log_determinant_ldlt.hpp>
#include <stan/math/prim/mat/fun/mdivide_left_ldlt.hpp>
#include <stan/math/prim/mat/fun/dot_product.hpp>
#include <stan/math/prim/mat/fun/mdivide_left_tri_low.hpp>
#include <stan/math/prim/mat/fun/multiply_lower_tri_self_transpose.hpp>
#include <stan/math/prim/mat/meta/index_type.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/prob/normal_rng.hpp>
#include <stan/math/prim/scal/prob/chi_square_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {

    template <class RNG>
    inline Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
    wishart_rng(const double nu,
                const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& S,
                RNG& rng) {
      using Eigen::MatrixXd;
      using stan::math::index_type;
      using stan::math::check_positive;
      using stan::math::check_size_match;
      using stan::math::check_square;

      static const char* function("stan::math::wishart_rng");

      typename index_type<MatrixXd>::type k = S.rows();

      check_positive(function, "degrees of freedom", nu);
      check_square(function, "scale parameter", S);

      MatrixXd B = MatrixXd::Zero(k, k);

      for (int j = 0; j < k; ++j) {
        for (int i = 0; i < j; ++i)
          B(i, j) = normal_rng(0, 1, rng);
        B(j, j) = std::sqrt(chi_square_rng(nu - j, rng));
      }

      return stan::math::crossprod(B * S.llt().matrixU());
    }


  }

}
#endif
