#ifndef STAN_MATH_PRIM_MAT_PROB_MULTI_STUDENT_T_LOG_HPP
#define STAN_MATH_PRIM_MAT_PROB_MULTI_STUDENT_T_LOG_HPP

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
#include <stan/math/prim/mat/prob/multi_normal_log.hpp>
#include <stan/math/prim/scal/meta/include_summand.hpp>
#include <stan/math/prim/scal/meta/VectorViewMvt.hpp>
#include <cmath>
#include <cstdlib>

namespace stan {

  namespace math {
    using Eigen::Dynamic;
    /**
     * Return the log of the multivariate Student t distribution
     * at the specified arguments.
     *
     * @tparam propto Carry out calculations up to a proportion
     */
    template <bool propto,
              typename T_y, typename T_dof, typename T_loc, typename T_scale>
    typename return_type<T_y, T_dof, T_loc, T_scale>::type
    multi_student_t_log(const T_y& y,
                        const T_dof& nu,
                        const T_loc& mu,
                        const T_scale& Sigma) {
      static const char* function("stan::math::multi_student_t");

      using stan::math::check_size_match;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_symmetric;
      using stan::math::check_positive;
      using boost::math::lgamma;
      using stan::math::log_determinant_ldlt;
      using stan::math::LDLT_factor;
      using stan::math::check_ldlt_factor;
      using stan::math::log1p;
      using std::log;

      typedef typename scalar_type<T_scale>::type T_scale_elem;
      typedef typename return_type<T_y, T_dof, T_loc, T_scale>::type lp_type;
      lp_type lp(0.0);

      // allows infinities
      check_not_nan(function, "Degrees of freedom parameter", nu);
      check_positive(function, "Degrees of freedom parameter", nu);

      using boost::math::isinf;

      if (isinf(nu))  // already checked nu > 0
        return multi_normal_log(y, mu, Sigma);

      using Eigen::Matrix;
      using std::vector;
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
                           "Size of one of the vectors of the random variable",
                           size_y_new,
                           "Size of another vector of the random variable",
                           size_y_old);
          size_y_old = size_y_new;
        }
        int size_mu_old = size_mu;
        int size_mu_new;
        for (size_t i = 1, size_ = length_mvt(mu); i < size_; i++) {
          int size_mu_new = mu_vec[i].size();
          check_size_match(function,
                           "Size of one of the vectors "
                           "of the location variable",
                           size_mu_new,
                           "Size of another vector of "
                           "the location variable",
                           size_mu_old);
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
                       "rows of scale parameter", Sigma.rows());
      check_size_match(function,
                       "Size of random variable", size_y,
                       "columns of scale parameter", Sigma.cols());

      for (size_t i = 0; i < size_vec; i++) {
        check_finite(function, "Location parameter", mu_vec[i]);
        check_not_nan(function, "Random variable", y_vec[i]);
      }
      check_symmetric(function, "Scale parameter", Sigma);


      LDLT_factor<T_scale_elem, Dynamic, Dynamic> ldlt_Sigma(Sigma);
      check_ldlt_factor(function, "LDLT_Factor of scale parameter", ldlt_Sigma);

      if (size_y == 0)  // y_vec[0].size() == 0
        return lp;

      if (include_summand<propto, T_dof>::value) {
        lp += lgamma(0.5 * (nu + size_y)) * size_vec;
        lp -= lgamma(0.5 * nu) * size_vec;
        lp -= (0.5 * size_y) * log(nu) * size_vec;
      }

      if (include_summand<propto>::value)
        lp -= (0.5 * size_y) * LOG_PI * size_vec;

      using stan::math::multiply;
      using stan::math::dot_product;
      using stan::math::subtract;
      using Eigen::Array;


      if (include_summand<propto, T_scale_elem>::value) {
        lp -= 0.5 * log_determinant_ldlt(ldlt_Sigma) * size_vec;
      }

      if (include_summand<propto, T_y, T_dof, T_loc, T_scale_elem>::value) {
        lp_type sum_lp_vec(0.0);
        for (size_t i = 0; i < size_vec; i++) {
          Eigen::Matrix<typename return_type<T_y, T_loc>::type, Dynamic, 1>
            y_minus_mu(size_y);
          for (int j = 0; j < size_y; j++)
            y_minus_mu(j) = y_vec[i](j)-mu_vec[i](j);
          sum_lp_vec += log1p(trace_inv_quad_form_ldlt(ldlt_Sigma, y_minus_mu)
                              / nu);
        }
        lp -= 0.5 * (nu + size_y) * sum_lp_vec;
      }
      return lp;
    }

    template <typename T_y, typename T_dof, typename T_loc, typename T_scale>
    inline
    typename return_type<T_y, T_dof, T_loc, T_scale>::type
    multi_student_t_log(const T_y& y, const T_dof& nu, const T_loc& mu,
                        const T_scale& Sigma) {
      return multi_student_t_log<false>(y, nu, mu, Sigma);
    }

  }
}
#endif
