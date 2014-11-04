#ifndef STAN__AGRAD__FWD__PROB__UNIVARIATE__CONTINUOUS__NORMAL_HPP
#define STAN__AGRAD__FWD__PROB__UNIVARIATE__CONTINUOUS__NORMAL_HPP

#include <boost/random/normal_distribution.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/utility/enable_if.hpp>
#include <stan/agrad/partials_vari.hpp>
#include <stan/math.hpp>
#include <stan/error_handling.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/constants.hpp>
#include <stan/prob/traits.hpp>

namespace stan {

  namespace prob {

    template <bool propto,
              typename T_y, typename T_loc, typename T_scale>
    typename boost::enable_if_c<contains_fvar<T_y,T_loc,T_scale>::value,
                                typename return_type<T_y,T_loc,T_scale>::type>::type
    normal_log(const T_y& y, const T_loc& mu, const T_scale& sigma) {
      // static const std::string& function("stan::prob::normal_log");
      // FIXME: add input checks

      using std::log;
      using stan::is_constant_struct;
      using stan::math::value_of;
      using stan::prob::include_summand;

      // check if no variables are involved and prop-to
      if (!include_summand<propto,T_y,T_loc,T_scale>::value)
        return 0.0;
      
      typename return_type<T_y,T_loc,T_scale>::type logp(0); 

      // log probability
      if (include_summand<propto>::value)
        logp += NEG_LOG_SQRT_TWO_PI;

      if (include_summand<propto,T_scale>::value)
        logp -= log(sigma);

      if (include_summand<propto,T_y,T_loc,T_scale>::value) {
        typename return_type<T_y,T_loc,T_scale>::type z = (y - mu) / sigma;
        logp -= 0.5 * z * z;
      }

      return logp;
    }

  }

}

#endif
