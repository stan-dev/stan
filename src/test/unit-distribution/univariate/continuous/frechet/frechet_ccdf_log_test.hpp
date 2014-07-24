// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/frechet.hpp>

#include <stan/math/functions/log1m_exp.hpp>
#include <stan/math/functions/multiply_log.hpp>
#include <stan/math/functions/value_of.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogFrechet : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);

    param[0] = 2.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.60653065));       // expected ccdf_log

    param[0] = 0.25;                // y
    param[1] = 2.9;                 // alpha
    param[2] = 1.8;                 // sigma
    parameters.push_back(param);
    ccdf_log.push_back(0);  // expected ccdf_log

    param[0] = 3.9;                 // y
    param[1] = 1.7;                 // alpha
    param[2] = 0.25;                // sigma
    parameters.push_back(param);
    ccdf_log.push_back(-4.675041342361127);  // expected ccdf_log
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());

    // alpha
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);
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

  template <typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  ccdf_log(const T_y& y, const T_shape& alpha, const T_scale& sigma,
           const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, 
           const T9&) {
    return stan::prob::frechet_ccdf_log(y, alpha, sigma);
  }
  
  template <typename T_y, typename T_shape, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  ccdf_log_function(const T_y& y, const T_shape& alpha, const T_scale& sigma,
               const T3&, const T4&, const T5&, const T6&, const T7&, 
               const T8&, const T9&) {
    using std::log;
    using std::pow;
	using std::exp;
    return log(1.0 - exp(-pow(sigma / y, alpha)));
  }
};
