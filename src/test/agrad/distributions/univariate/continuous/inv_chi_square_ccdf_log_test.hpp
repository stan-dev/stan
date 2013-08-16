// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/inv_chi_square.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogInvChiSquare : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(2);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // Degrees of freedom
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.317528));  // expected ccdf_log

  }
  
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {

    // y
    index.push_back(0U);
    value.push_back(-1.0);
 
    // nu
    index.push_back(1U);
    value.push_back(-1.0);
      
    index.push_back(1U);
    value.push_back(0.0);
      
    index.push_back(1U);
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
    
  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof>::type 
  ccdf_log(const T_y& y, const T_dof& nu, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::inv_chi_square_ccdf_log(y, nu);
  }


  
  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof>::type 
  ccdf_log_function(const T_y& y, const T_dof& nu, const T2&,
                    const T3&, const T4&, const T5&, const T6&, 
                    const T7&, const T8&, const T9&) {
    using stan::agrad::gamma_q;
    using stan::math::gamma_q;
    
    return log(1.0 - gamma_q(0.5 * nu, 0.5 / y));  
  }
};
