// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/inv_gamma_ccdf_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCcdfLogInvGamma : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // alpha (Shape)
    param[2] = 3.3;           // beta (Scale)
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.138010737568659559638900550956));  // expected ccdf_log

  }
  
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
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
    return false;
  }
    
  template <typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  ccdf_log(const T_y& y, const T_shape& alpha, const T_scale& beta,
          const T3&, const T4&, const T5&) {
    return stan::math::inv_gamma_ccdf_log(y, alpha, beta);
  }


  
  template <typename T_y, typename T_shape, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_scale>::type 
  ccdf_log_function(const T_y& y, const T_shape& alpha, const T_scale& beta,
                    const T3&, const T4&, const T5&) {
    using stan::math::gamma_q;
    using stan::math::gamma_q;

    return log(1.0 - gamma_q(alpha, beta / y));  }
    
};
