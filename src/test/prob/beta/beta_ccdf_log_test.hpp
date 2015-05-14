// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/beta_ccdf_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCcdfLogBeta : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_ccdf) {
    vector<double> param(3);

    param[0] = 0.5;           // y
    param[1] = 2.0;           // alpha (Success Scale)
    param[2] = 5.0;           // beta  (Faiulre Scale)
    parameters.push_back(param);
    log_ccdf.push_back(std::log(1.0 - 0.890625));  // expected Log_CCDF

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
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale_succ, T_scale_fail>::type 
  ccdf_log(const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::beta_ccdf_log(y, alpha, beta);
  }
  
  template <typename T_y, typename T_scale_succ, typename T_scale_fail,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_scale_succ, T_scale_fail>::type 
  ccdf_log_function(const T_y& y, const T_scale_succ& alpha, 
                    const T_scale_fail& beta, const T3&, const T4&, const T5&) {
    return stan::math::beta_ccdf_log(y, alpha, beta);
      
  }
    
};
