// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/exponential.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfLogExponential : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf_log) {
    vector<double> param(2);

    param[0] = 2.0;                 // y
    param[1] = 1.5;                 // beta
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.95021293163213605702));  // expected cdf_log

    param[0] = 15.0;                // y
    param[1] = 3.9;                 // beta
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.999999999999999999999));  // expected cdf_log
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());
    
    // beta
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
  }

  bool has_lower_bound() {
    return true;
  }

  double lower_bound() {
    return 0.0;
  }
  
  bool has_upper_bound() {
    return false;
  }

  template <typename T_y, typename T_inv_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_inv_scale>::type 
  cdf_log(const T_y& y, const T_inv_scale& beta, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exponential_cdf_log(y, beta);
  }
  
  
  template <typename T_y, typename T_inv_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_inv_scale>::type 
  cdf_log_function(const T_y& y, const T_inv_scale& beta, 
        const T2&, const T3&, const T4&, const T5&, 
        const T6&, const T7&, const T8&, const T9&) {
    using std::log;
    using std::exp;

    return log(1.0 - exp(-beta * y));
  }
};
