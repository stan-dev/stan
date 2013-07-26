#ifndef __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__RAYLEIGH_HPP__
#define __STAN__PROB__DISTRIBUTIONS__UNIVARIATE__CONTINUOUS__RAYLEIGH_HPP__

#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>

#include <stan/agrad.hpp>
#include <stan/math.hpp>
#include <stan/math/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    template <bool propto, 
              typename T_y, typename T_scale>
    typename return_type<T_y,T_scale>::type
    rayleigh_log(const T_y& y, const T_scale& sigma) {
      static const char* function = "stan::prob::rayleigh_log(%1%)";

      using std::log;
      using stan::is_constant_struct;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if any vectors are zero length
      if (!(stan::length(y) && stan::length(sigma)))
        return 0.0;

      // set up return value accumulator
      double logp(0.0);

      // validate args (here done over var, which should be OK)
      if (!check_not_nan(function, y, "Random variable", &logp))
        return logp;
      if (!check_positive(function, sigma, "Scale parameter", &logp))
        return logp;
      if (!check_positive(function, y, "Random variable", &logp))
        return logp;
      if (!(check_consistent_sizes(function,
                                   y,sigma,
                                   "Random variable","Scale parameter",
                                   &logp)))
        return logp;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_scale>::value)
        return 0.0;
      
      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_scale> operands_and_partials(y, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, sigma);

      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      DoubleVectorView<include_summand<propto,T_scale>::value,
                       is_vector<T_scale>::value> log_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
        if (include_summand<propto,T_scale>::value)
          log_sigma[i] = log(value_of(sigma_vec[i]));
      }

      for (size_t n = 0; n < N; n++) {
        // pull out values of arguments
        const double y_dbl = value_of(y_vec[n]);
      
        // reusable subexpression values
        const double y_over_sigma = y_dbl * inv_sigma[n];

        static double NEGATIVE_HALF = -0.5;

        // log probability
        if (include_summand<propto,T_scale>::value)
          logp -= 2.0 * log_sigma[n];        
        if (include_summand<propto,T_y>::value)
          logp += log(y_dbl);
        // if (include_summand<propto,T_y,T_scale>::value)
        logp += NEGATIVE_HALF * y_over_sigma * y_over_sigma;

        // gradients
        double scaled_diff = inv_sigma[n] * y_over_sigma;
        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += 1.0 / y_dbl - scaled_diff;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] 
            += y_over_sigma * scaled_diff - 2.0 * inv_sigma[n];
      }
      return operands_and_partials.to_var(logp);
    }

    template <typename T_y, typename T_scale>
    inline
    typename return_type<T_y,T_scale>::type
    rayleigh_log(const T_y& y, const T_scale& sigma) {
      return rayleigh_log<false>(y,sigma);
    }

    template <typename T_y, typename T_scale>
    typename return_type<T_y,T_scale>::type
    rayleigh_cdf(const T_y& y, const T_scale& sigma) {
      static const char* function = "stan::prob::rayleigh_cdf(%1%)";

      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      using stan::is_constant_struct;
      using stan::math::square;
      using stan::math::value_of;

      double cdf(1.0);

      // check if any vectors are zero length
      if (!(stan::length(y) && stan::length(sigma)))
        return cdf;

      if (!check_not_nan(function, y, "Random variable", &cdf))
        return cdf;
      if (!check_nonnegative(function, y, "Random variable", &cdf))
        return cdf;
      if (!check_not_nan(function, sigma, "Scale parameter", &cdf))
        return cdf;
      if (!check_positive(function, sigma, "Scale parameter", &cdf))
        return cdf;
      if (!(check_consistent_sizes(function,
                                   y,sigma,
                                   "Random variable","Scale parameter",
                                   &cdf)))
        return cdf;


      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_scale> operands_and_partials(y, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, sigma);
      
      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
      }

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double y_sqr = y_dbl * y_dbl;
        const double inv_sigma_sqr = inv_sigma[n] * inv_sigma[n];
        const double exp_val = exp(-0.5 * y_sqr * inv_sigma_sqr);

        if (include_summand<false,T_y,T_scale>::value)
          cdf *= (1.0 - exp_val);
      }

      //gradients
      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double y_sqr = square(y_dbl);
        const double inv_sigma_sqr = square(inv_sigma[n]);
        const double exp_val = exp(-0.5 * y_sqr * inv_sigma_sqr);
        const double exp_div_1m_exp = exp_val / (1.0 - exp_val);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += y_dbl * inv_sigma_sqr 
            * exp_div_1m_exp * cdf;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] -= y_sqr * inv_sigma_sqr
            * inv_sigma[n] * exp_div_1m_exp * cdf;
      }

      return operands_and_partials.to_var(cdf);
    }

    template <typename T_y, typename T_scale>
    typename return_type<T_y,T_scale>::type
    rayleigh_cdf_log(const T_y& y, const T_scale& sigma) {
      static const char* function = "stan::prob::rayleigh_cdf_log(%1%)";

      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      using stan::is_constant_struct;
      using stan::math::square;
      using stan::math::value_of;

      double cdf_log(0.0);

      // check if any vectors are zero length
      if (!(stan::length(y) && stan::length(sigma)))
        return cdf_log;

      if (!check_not_nan(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if (!check_nonnegative(function, y, "Random variable", &cdf_log))
        return cdf_log;
      if (!check_not_nan(function, sigma, "Scale parameter", &cdf_log))
        return cdf_log;
      if (!check_positive(function, sigma, "Scale parameter", &cdf_log))
        return cdf_log;
      if (!(check_consistent_sizes(function,
                                   y,sigma,
                                   "Random variable","Scale parameter",
                                   &cdf_log)))
        return cdf_log;


      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_scale> operands_and_partials(y, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, sigma);
      
      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
      }

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double y_sqr = y_dbl * y_dbl;
        const double inv_sigma_sqr = inv_sigma[n] * inv_sigma[n];
        const double exp_val = exp(-0.5 * y_sqr * inv_sigma_sqr);
        const double exp_div_1m_exp = exp_val / (1.0 - exp_val);

        if (include_summand<false,T_y,T_scale>::value)
          cdf_log += log(1.0 - exp_val);

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] += y_dbl * inv_sigma_sqr 
            * exp_div_1m_exp;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] -= y_sqr * inv_sigma_sqr
            * inv_sigma[n] * exp_div_1m_exp;
      }

      return operands_and_partials.to_var(cdf_log);
    }

    template <typename T_y, typename T_scale>
    typename return_type<T_y,T_scale>::type
    rayleigh_ccdf_log(const T_y& y, const T_scale& sigma) {
      static const char* function = "stan::prob::rayleigh_ccdf_log(%1%)";

      using stan::math::check_nonnegative;
      using stan::math::check_positive;
      using stan::math::check_finite;
      using stan::math::check_not_nan;
      using stan::math::check_consistent_sizes;
      using stan::prob::include_summand;
      using stan::is_constant_struct;
      using stan::math::square;
      using stan::math::value_of;

      double ccdf_log(0.0);

      // check if any vectors are zero length
      if (!(stan::length(y) && stan::length(sigma)))
        return ccdf_log;

      if (!check_not_nan(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if (!check_nonnegative(function, y, "Random variable", &ccdf_log))
        return ccdf_log;
      if (!check_not_nan(function, sigma, "Scale parameter", &ccdf_log))
        return ccdf_log;
      if (!check_positive(function, sigma, "Scale parameter", &ccdf_log))
        return ccdf_log;
      if (!(check_consistent_sizes(function,
                                   y,sigma,
                                   "Random variable","Scale parameter",
                                   &ccdf_log)))
        return ccdf_log;


      // set up template expressions wrapping scalars into vector views
      agrad::OperandsAndPartials<T_y, T_scale> operands_and_partials(y, sigma);

      VectorView<const T_y> y_vec(y);
      VectorView<const T_scale> sigma_vec(sigma);
      size_t N = max_size(y, sigma);
      
      DoubleVectorView<true,is_vector<T_scale>::value> inv_sigma(length(sigma));
      for (size_t i = 0; i < length(sigma); i++) {
        inv_sigma[i] = 1.0 / value_of(sigma_vec[i]);
      }

      for (size_t n = 0; n < N; n++) {
        const double y_dbl = value_of(y_vec[n]);
        const double y_sqr = y_dbl * y_dbl;
        const double inv_sigma_sqr = inv_sigma[n] * inv_sigma[n];

        if (include_summand<false,T_y,T_scale>::value)
          ccdf_log += -0.5 * y_sqr * inv_sigma_sqr;

        if (!is_constant_struct<T_y>::value)
          operands_and_partials.d_x1[n] -= y_dbl * inv_sigma_sqr;
        if (!is_constant_struct<T_scale>::value)
          operands_and_partials.d_x2[n] += y_sqr * inv_sigma_sqr
            * inv_sigma[n];
      }

      return operands_and_partials.to_var(ccdf_log);
    }

    template <class RNG>
    inline double
    rayleigh_rng(const double sigma,
                 RNG& rng) {
      using boost::variate_generator;
      using boost::random::uniform_real_distribution;
      variate_generator<RNG&, uniform_real_distribution<> >
        uniform_rng(rng, uniform_real_distribution<>(0.0, 1.0));
      return sigma * std::sqrt(-2.0 * std::log(uniform_rng()));
    }
  }
}
#endif
