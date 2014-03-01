// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/chi_square.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogChiSquare : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(2);

    param[0] = 7.9;                 // y
    param[1] = 3.0;                 // nu
    parameters.push_back(param);
    ccdf_log.push_back(-3.0339689);  // expected ccdf_log

    param[0] = 1.9;                 // y
    param[1] = 0.5;                 // nu
    parameters.push_back(param);
    ccdf_log.push_back(-2.6142213);    // expected ccdf_log
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);
    
    // nu
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

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
  typename stan::return_type<T_y, T_dof, T2>::type 
  ccdf_log(const T_y& y, const T_dof& nu, 
     const T2&, const T3&, const T4&, const T5&, 
     const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::chi_square_ccdf_log(y, nu);
  }

  template <typename T_y, typename T_dof, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_dof, T2>::type 
  ccdf_log_function(const T_y& y, const T_dof& nu, 
      const T2&, const T3&, const T4&, const T5&, 
      const T6&, const T7&, const T8&, const T9&) {
    using stan::agrad::gamma_p;
    using stan::math::gamma_p;
    using std::log;

    return log(1.0 - gamma_p(nu * 0.5, y * 0.5));
  }
};
