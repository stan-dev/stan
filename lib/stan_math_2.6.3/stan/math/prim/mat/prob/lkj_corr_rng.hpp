#ifndef STAN_MATH_PRIM_MAT_PROB_LKJ_CORR_RNG_HPP
#define STAN_MATH_PRIM_MAT_PROB_LKJ_CORR_RNG_HPP

#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/mat/fun/factor_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/factor_U.hpp>
#include <stan/math/prim/mat/fun/read_corr_L.hpp>
#include <stan/math/prim/mat/fun/read_corr_matrix.hpp>
#include <stan/math/prim/mat/fun/read_cov_L.hpp>
#include <stan/math/prim/mat/fun/read_cov_matrix.hpp>
#include <stan/math/prim/mat/fun/make_nu.hpp>
#include <stan/math/prim/scal/fun/identity_constrain.hpp>
#include <stan/math/prim/scal/fun/identity_free.hpp>
#include <stan/math/prim/scal/fun/positive_constrain.hpp>
#include <stan/math/prim/scal/fun/positive_free.hpp>
#include <stan/math/prim/scal/fun/lb_constrain.hpp>
#include <stan/math/prim/scal/fun/lb_free.hpp>
#include <stan/math/prim/scal/fun/ub_constrain.hpp>
#include <stan/math/prim/scal/fun/ub_free.hpp>
#include <stan/math/prim/scal/fun/lub_constrain.hpp>
#include <stan/math/prim/scal/fun/lub_free.hpp>
#include <stan/math/prim/scal/fun/prob_constrain.hpp>
#include <stan/math/prim/scal/fun/prob_free.hpp>
#include <stan/math/prim/scal/fun/corr_constrain.hpp>
#include <stan/math/prim/scal/fun/corr_free.hpp>
#include <stan/math/prim/mat/fun/unit_vector_constrain.hpp>
#include <stan/math/prim/mat/fun/unit_vector_free.hpp>
#include <stan/math/prim/mat/fun/simplex_constrain.hpp>
#include <stan/math/prim/mat/fun/simplex_free.hpp>
#include <stan/math/prim/mat/fun/ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/ordered_free.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_constrain.hpp>
#include <stan/math/prim/mat/fun/positive_ordered_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_factor_free.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_constrain.hpp>
#include <stan/math/prim/mat/fun/cholesky_corr_free.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/corr_matrix_free.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_constrain_lkj.hpp>
#include <stan/math/prim/mat/fun/cov_matrix_free_lkj.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_rng.hpp>

namespace stan {
  namespace math {

    template <class RNG>
    inline Eigen::MatrixXd
    lkj_corr_rng(const size_t K,
                 const double eta,
                 RNG& rng) {
      static const char* function("stan::math::lkj_corr_rng");

      using stan::math::check_positive;

      check_positive(function, "Shape parameter", eta);

      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(lkj_corr_cholesky_rng(K, eta,
                                                                     rng));
    }

  }
}
#endif
