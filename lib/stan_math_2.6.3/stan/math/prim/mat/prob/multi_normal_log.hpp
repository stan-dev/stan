#ifndef STAN_MATH_PRIM_MAT_PROB_MULTI_NORMAL_LOG_HPP
#define STAN_MATH_PRIM_MAT_PROB_MULTI_NORMAL_LOG_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <stan/math/prim/mat/err/check_ldlt_factor.hpp>
#include <stan/math/prim/scal/err/check_size_match.hpp>
#include <stan/math/prim/mat/err/check_symmetric.hpp>
#include <stan/math/prim/scal/err/check_finite.hpp>
#include <stan/math/prim/scal/err/check_not_nan.hpp>
#include <stan/math/prim/scal/err/check_positive.hpp>
#include <stan/math/prim/mat/fun/trace_inv_quad_form_ldlt.hpp>
#include <stan/math/prim/mat/fun/log_determinant_ldlt.hpp>
#include <stan/math/prim/scal/meta/return_type.hpp>
#include <stan/math/prim/scal/meta/VectorViewMvt.hpp>
#include <stan/math/prim/scal/meta/max_size_mvt.hpp>
#include <stan/math/prim/scal/fun/constants.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>

namespace stan {

  namespace math {
    using Eigen::Dynamic;

    template <bool propto,
              typename T_y, typename T_loc, typename T_covar>
    typename return_type<T_y, T_loc, T_covar>::type
    multi_normal_log(const T_y& y,
                     const T_loc& mu,
                     const T_covar& Sigma) {
      static const char* function("stan::math::multi_normal_log");
      typedef typename scalar_type<T_covar>::type T_covar_elem;
      typedef typename return_type<T_y, T_loc, T_covar>::type lp_type;
      lp_type lp(0.0);

      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_positive;
      using stan::math::check_symmetric;
      using stan::math::check_ldlt_factor;

      check_positive(function, "Covariance matrix rows", Sigma.rows());
      check_symmetric(function, "Covariance matrix", Sigma);

      stan::math::LDLT_factor<T_covar_elem, Dynamic, Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function,
                        "LDLT_Factor of covariance parameter", ldlt_Sigma);

      VectorViewMvt<const T_y> y_vec(y);
      VectorViewMvt<const T_loc> mu_vec(mu);
      // size of std::vector of Eigen vectors
      size_t size_vec = max_size_mvt(y, mu);

      // Check if every vector of the array has the same size
      int size_y = y_vec[0].size();
      int size_mu = mu_vec[0].size();
      if (size_vec > 1) {
        int size_y_old = size_y;
        int size_y_new;
        for (size_t i = 1, size_ = length_mvt(y); i < size_; i++) {
          int size_y_new = y_vec[i].size();
          check_size_match(function,
                           "Size of one of the vectors of "
                           "the random variable", size_y_new,
                           "Size of another vector of the "
                           "random variable", size_y_old);
          size_y_old = size_y_new;
        }
        int size_mu_old = size_mu;
        int size_mu_new;
        for (size_t i = 1, size_ = length_mvt(mu); i < size_; i++) {
          int size_mu_new = mu_vec[i].size();
          check_size_match(function,
                           "Size of one of the vectors of "
                           "the location variable", size_mu_new,
                           "Size of another vector of the "
                           "location variable", size_mu_old);
          size_mu_old = size_mu_new;
        }
        (void) size_y_old;
        (void) size_y_new;
        (void) size_mu_old;
        (void) size_mu_new;
      }

      check_size_match(function,
                       "Size of random variable", size_y,
                       "size of location parameter", size_mu);
      check_size_match(function,
                       "Size of random variable", size_y,
                       "rows of covariance parameter", Sigma.rows());
      check_size_match(function,
                       "Size of random variable", size_y,
                       "columns of covariance parameter", Sigma.cols());

      for (size_t i = 0; i < size_vec; i++) {
        check_finite(function, "Location parameter", mu_vec[i]);
        check_not_nan(function, "Random variable", y_vec[i]);
      }

      if (size_y == 0)  // y_vec[0].size() == 0
        return lp;

      if (include_summand<propto>::value)
        lp += NEG_LOG_SQRT_TWO_PI * size_y * size_vec;

      if (include_summand<propto, T_covar_elem>::value)
        lp -= 0.5 * log_determinant_ldlt(ldlt_Sigma) * size_vec;

      if (include_summand<propto, T_y, T_loc, T_covar_elem>::value) {
        lp_type sum_lp_vec(0.0);
        for (size_t i = 0; i < size_vec; i++) {
          Eigen::Matrix<typename return_type<T_y, T_loc>::type, Dynamic, 1>
            y_minus_mu(size_y);
          for (int j = 0; j < size_y; j++)
            y_minus_mu(j) = y_vec[i](j)-mu_vec[i](j);
          sum_lp_vec += trace_inv_quad_form_ldlt(ldlt_Sigma, y_minus_mu);
        }
        lp -= 0.5*sum_lp_vec;
      }
      return lp;
    }

    template <typename T_y, typename T_loc, typename T_covar>
    inline
    typename return_type<T_y, T_loc, T_covar>::type
    multi_normal_log(const T_y& y,
                     const T_loc& mu,
                     const T_covar& Sigma) {
      return multi_normal_log<false>(y, mu, Sigma);
    }

  }
}

#endif
