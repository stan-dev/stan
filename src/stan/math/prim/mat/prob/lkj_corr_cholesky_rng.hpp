#ifndef STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_CHOLESKY_RNG_HPP
#define STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_CHOLESKY_RNG_HPP

#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/beta_rng.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/fun/factor_cov_matrix.hpp>
#include <stan/math/prim/scal/fun/factor_U.hpp>
#include <stan/math/prim/scal/fun/read_corr_L.hpp>
#include <stan/math/prim/scal/fun/read_corr_matrix.hpp>
#include <stan/math/prim/scal/fun/read_cov_L.hpp>
#include <stan/math/prim/scal/fun/read_cov_matrix.hpp>
#include <stan/math/prim/scal/fun/make_nu.hpp>
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
#include <stan/math/prim/scal/fun/unit_vector_constrain.hpp>
#include <stan/math/prim/scal/fun/unit_vector_free.hpp>
#include <stan/math/prim/scal/fun/simplex_constrain.hpp>
#include <stan/math/prim/scal/fun/simplex_free.hpp>
#include <stan/math/prim/scal/fun/ordered_constrain.hpp>
#include <stan/math/prim/scal/fun/ordered_free.hpp>
#include <stan/math/prim/scal/fun/positive_ordered_constrain.hpp>
#include <stan/math/prim/scal/fun/positive_ordered_free.hpp>
#include <stan/math/prim/scal/fun/cholesky_factor_constrain.hpp>
#include <stan/math/prim/scal/fun/cholesky_factor_free.hpp>
#include <stan/math/prim/scal/fun/cholesky_corr_constrain.hpp>
#include <stan/math/prim/scal/fun/cholesky_corr_free.hpp>
#include <stan/math/prim/scal/fun/corr_matrix_constrain.hpp>
#include <stan/math/prim/scal/fun/corr_matrix_free.hpp>
#include <stan/math/prim/scal/fun/cov_matrix_constrain.hpp>
#include <stan/math/prim/scal/fun/cov_matrix_free.hpp>
#include <stan/math/prim/scal/fun/cov_matrix_constrain_lkj.hpp>
#include <stan/math/prim/scal/fun/cov_matrix_free_lkj.hpp>

namespace stan {
  namespace prob {

    template <class RNG>
    inline Eigen::MatrixXd
    lkj_corr_cholesky_rng(const size_t K,
                          const double eta,
                          RNG& rng) {
      static const char* function("stan::prob::lkj_corr_cholesky_rng");

      using stan::math::check_positive;
      
      check_positive(function, "Shape parameter", eta);

      Eigen::ArrayXd CPCs( (K * (K - 1)) / 2 );
      double alpha = eta + 0.5 * (K - 1);
      unsigned int count = 0;
      for (size_t i = 0; i < (K - 1); i++) {
        alpha -= 0.5;
        for (size_t j = i + 1; j < K; j++) {
          CPCs(count) = 2.0 * stan::prob::beta_rng(alpha,alpha,rng) - 1.0;
          count++;
        }
      }
      return stan::prob::read_corr_L(CPCs, K);
    }

  }
}
#endif
