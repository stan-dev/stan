#ifndef STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_RNG_HPP
#define STAN__MATH__PRIM__MAT__PROB__LKJ_CORR_RNG_HPP

#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/scal/meta/constants.hpp>
#include <stan/math/prim/scal/prob/beta.hpp>
#include <stan/math/prim/scal/meta/prob_traits.hpp>
#include <stan/math/prim/scal/fun/transform.hpp>
#include <stan/math/prim/mat/prob/lkj_corr_cholesky_rng.hpp>

namespace stan {
  namespace prob {

    template <class RNG>
    inline Eigen::MatrixXd
    lkj_corr_rng(const size_t K,
                 const double eta,
                 RNG& rng) {

      static const char* function("stan::prob::lkj_corr_rng");

      using stan::math::check_positive;
      
      check_positive(function, "Shape parameter", eta);

      using stan::math::multiply_lower_tri_self_transpose;
      return multiply_lower_tri_self_transpose(lkj_corr_cholesky_rng(K, eta, rng));
    }

  }
}
#endif
