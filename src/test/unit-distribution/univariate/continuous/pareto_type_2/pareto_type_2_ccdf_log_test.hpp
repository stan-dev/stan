// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/pareto_type_2.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogParetoType2 : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(4);

    param[0] = 0.1;           // y
    param[1] = 0;           // mu
    param[2] = 0.5;           // lambda
    param[3] = 3.0;           // alpha
    parameters.push_back(param);
    ccdf_log.push_back(std::log(0.5787037037037038311738));  // expected CCDF_log

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
    // y_min
    index.push_back(2U);
    value.push_back(-1.0);
      
    index.push_back(2U);
    value.push_back(0.0);
      
    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());
      
    // alpha
    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(0.0);

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());

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
    
  template <typename T_y, typename T_loc, typename T_scale, typename T_shape,
      typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  ccdf_log(const T_y& y, const T_loc& mu, const T_scale& lambda, const T_shape& alpha,
           const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::pareto_type_2_ccdf_log(y, mu, lambda, alpha);
  }
  
  template <typename T_y, typename T_loc, typename T_scale, typename T_shape,
      typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  ccdf_log_function(const T_y& y, const T_loc& mu, const T_scale& lambda, const T_shape& alpha,
         const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      using std::log;
      using std::pow;
      return log(pow(1.0 + (y-mu)/lambda,-alpha));
  }
    
};
