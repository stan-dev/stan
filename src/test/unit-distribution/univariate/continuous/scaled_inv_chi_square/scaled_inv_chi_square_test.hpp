// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/scaled_inv_chi_square.hpp>

#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsScaledInvChiSquare : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 12.7;          // y
    param[1] = 6.1;           // nu
    param[2] = 3.0;           // s
    parameters.push_back(param);
    log_prob.push_back(-3.091965); // expected log_prob

    param[0] = 1.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.5;           // s
    parameters.push_back(param);
    log_prob.push_back(-1.737086); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // nu
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // s
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }


  template <class T_y, class T_dof, class T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob(const T_y& y, const T_dof& nu, const T_scale& s,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::scaled_inv_chi_square_log(y, nu, s);
  }

  template <bool propto, 
      class T_y, class T_dof, class T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  log_prob(const T_y& y, const T_dof& nu, const T_scale& s,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::scaled_inv_chi_square_log<propto>(y, nu, s);
  }
  
  
  template <class T_y, class T_dof, class T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_dof& nu, const T_scale& s,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using stan::prob::include_summand;
    using stan::math::multiply_log;
    using stan::math::square;

    
    if (y <= 0)
      return stan::prob::LOG_ZERO;
    
    var logp(0);
    if (include_summand<true,T_dof>::value) {
      var half_nu = 0.5 * nu;
      logp += multiply_log(half_nu,half_nu) - lgamma(half_nu);
    }
    if (include_summand<true,T_dof,T_scale>::value)
      logp += nu * log(s);
    if (include_summand<true,T_dof,T_y>::value)
      logp -= multiply_log(nu*0.5+1.0, y);
    if (include_summand<true,T_dof,T_y,T_scale>::value)
      logp -= nu * 0.5 * square(s) / y;
    return logp;
  }
};
