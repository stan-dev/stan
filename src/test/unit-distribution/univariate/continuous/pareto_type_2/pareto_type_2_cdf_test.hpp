// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/pareto_type_2.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfParetoType2 : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 0.1;           // y
    param[1] = 0;           // mu
    param[2] = 0.5;           // lambda
    param[3] = 3.0;           // alpha
    parameters.push_back(param);
    cdf.push_back(0.4212962962962961688262);  // expected CDF

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
      typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  cdf(const T_y& y, const T_loc& mu, const T_scale& lambda, const T_shape& alpha,
      const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::pareto_type_2_cdf(y, mu, lambda, alpha);
  }
  
  template <typename T_y, typename T_loc, typename T_scale, typename T_shape,
     typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale, T_shape>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& lambda, const T_shape& alpha,
         const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
      using std::pow;
      return 1.0 - pow(1.0 + (y-mu)/lambda,-alpha);
  }
    
};
