// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/von_mises.hpp>

#include <stan/math/constants.hpp>
#include <stan/math/functions/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionVonMises : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = boost::math::constants::third_pi<double>(); // y
    param[1] = boost::math::constants::sixth_pi<double>(); // mu
    param[2] = 0.5;                                         // kappa
    parameters.push_back(param);
    log_prob.push_back(-1.4664141);

    param[0] = -boost::math::constants::sixth_pi<double>();
    param[1] = -boost::math::constants::three_quarters_pi<double>();
    param[2] = 1.0;
    parameters.push_back(param);
    log_prob.push_back(-2.33261);

    param[0] = boost::math::constants::pi<double>() / 4.;
    param[1] = -boost::math::constants::three_quarters_pi<double>();
    param[2] = 1.5;
    parameters.push_back(param);
    log_prob.push_back(-3.8366644);

    param[0] = -boost::math::constants::sixth_pi<double>();
    param[1] = boost::math::constants::sixth_pi<double>();
    param[2] = 4.0;
    parameters.push_back(param);
    log_prob.push_back(-2.26285);
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(0U);
    value.push_back(numeric_limits<double>::infinity());

    // mu
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    // kappa
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& kappa,
           const T3&, const T4&, const T5&, const T6&, const T7&, 
           const T8&, const T9&) {
    return stan::prob::von_mises_log(y, mu, kappa);
  }

  template <bool propto, 
      typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  log_prob(const T_y& y, const T_loc& mu, const T_scale& kappa,
           const T3&, const T4&, const T5&, const T6&, const T7&, 
           const T8&, const T9&) {
    return stan::prob::von_mises_log<propto>(y, mu, kappa);
  }
  
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  var log_prob_function(const T_y& y, const T_loc& mu, const T_scale& kappa,
                        const T3&, const T4&, const T5&, const T6&, const T7&,
                        const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::pi;
    using stan::math::modified_bessel_first_kind;
    using std::log;

    var lp(0.0);

    if (include_summand<true>::value) 
      lp -= log(2.0 * stan::math::pi());
    if (include_summand<true,T_scale>::value)
      lp -= log(modified_bessel_first_kind(0,kappa));
    if (include_summand<true,T_y,T_loc,T_scale>::value)
      lp += kappa * cos(mu - y);

    return lp;
  }
};

