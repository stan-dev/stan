#ifndef STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_CHOLESKY_RNG_HPP
#define STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_CHOLESKY_RNG_HPP

#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/beta_rng.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>
#include <stan/math/prim/scal/fun/transform.hpp>

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
