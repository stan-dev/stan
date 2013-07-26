// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/logistic.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogLogistic : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // Location
    param[2] = 3.3;           // Scale
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.680827173));  // expected ccdf_log

  }
  
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
 
    // mu
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());
      
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());
      
    // sigma
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }
    
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::logistic_ccdf_log(y, mu, sigma);
  }


  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  ccdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return log(1.0 - 1.0 / (1.0 + exp(-(y - mu) / sigma)));
  }
    
};
