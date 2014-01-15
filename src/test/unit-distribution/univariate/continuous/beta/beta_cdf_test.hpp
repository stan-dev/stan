// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/beta.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfBeta : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 0.5;           // y
    param[1] = 4.4;           // alpha (Success Scale)
    param[2] = 3.2;           // beta  (Faiulre Scale)
    parameters.push_back(param);
    cdf.push_back(0.3223064740892);  // expected CDF

  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
    
    index.push_back(0U);
    value.push_back(2.0);

    // alpha
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());
      
    // beta
    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

  }
  
  bool has_lower_bound() {
    return true;
  }

  double lower_bound() {
    return 0.0;
  }
    
  bool has_upper_bound() {
    return true;
  }

  double upper_bound() {
    return 1.0;
  }
    
  template <typename T_y, typename T_scale_succ, typename T_scale_fail,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale_succ, T_scale_fail>::type 
  cdf(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_cdf(y, alpha, beta);
  }


  
  template <typename T_y, typename T_scale_succ, typename T_scale_fail,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale_succ, T_scale_fail>::type 
  cdf_function(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::beta_cdf(y, alpha, beta);
      
  }
    
};
