// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/inv_chi_square.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsInvChiSquare : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(2);

    param[0] = 0.5;                 // y
    param[1] = 2.0;                 // nu
    parameters.push_back(param);
    log_prob.push_back(-0.3068528);  // expected log_prob

    param[0] = 3.2;                 // y
    param[1] = 9.1;                 // nu
    parameters.push_back(param);
    log_prob.push_back(-12.28905);    // expected log_prob
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
  }

  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof>::type 
  log_prob(const T_y& y, const T_dof& nu, 
       const T2&, const T3&, const T4&, const T5&,
       const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::inv_chi_square_log(y, nu);
  }

  template <bool propto, 
      typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof>::type 
  log_prob(const T_y& y, const T_dof& nu, 
       const T2&, const T3&, const T4&, const T5&, 
       const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::inv_chi_square_log<propto>(y, nu);
  }
  
  
  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_y& y, const T_dof& nu, 
        const T2&, const T3&, const T4&, const T5&, 
        const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::multiply_log;
    using boost::math::lgamma;
    using stan::prob::NEG_LOG_TWO_OVER_TWO;
    
    var logp(0);
    if (include_summand<true,T_dof>::value)
      logp += nu * NEG_LOG_TWO_OVER_TWO - lgamma(0.5 * nu);
    if (include_summand<true,T_y,T_dof>::value)
      logp -= multiply_log(0.5*nu+1.0, y);
    if (include_summand<true,T_y>::value)
      logp -= 0.5 / y;
    return logp;
  }
};


