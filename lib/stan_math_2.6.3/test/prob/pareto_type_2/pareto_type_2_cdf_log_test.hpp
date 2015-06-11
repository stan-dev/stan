// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/pareto_type_2_cdf_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfLogParetoType2 : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf_log) {
    vector<double> param(4);

    param[0] = 0.1;           // y
    param[1] = 0;           // mu
    param[2] = 0.5;           // lambda
    param[3] = 3.0;           // alpha
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.4212962962962961688262));  // expected CDF_log

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
    // lambda
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
            typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  cdf_log(const T_y& y, const T_loc& mu, const T_scale& lambda, 
          const T_shape& alpha, const T4&, const T5&) {
    return stan::math::pareto_type_2_cdf_log(y, mu, lambda, alpha);
  }
  
  template <typename T_y, typename T_loc, typename T_scale, typename T_shape,
            typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  cdf_log_function(const T_y& y, const T_loc& mu, const T_scale& lambda, 
                   const T_shape& alpha, const T4&, const T5&) {
    using stan::math::log1m;
    using std::pow;
    return log1m(pow(1.0 + (y-mu)/lambda,-alpha));
  }
    
};
