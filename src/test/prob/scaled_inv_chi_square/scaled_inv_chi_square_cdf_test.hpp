// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/scaled_inv_chi_square_cdf.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfScaledInvChiSquare : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 3.0;           // y
    param[1] = 0.5;           // nu (Degrees of Freedom)
    param[2] = 3.3;           // s  (Scale)
    parameters.push_back(param);
    cdf.push_back(0.078121091257371137070194);  // expected CDF
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
      
    // s
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
    
  template <typename T_y, typename T_dof, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  cdf(const T_y& y, const T_dof& nu, const T_scale& s,
      const T3&, const T4&, const T5&) {
    return stan::math::scaled_inv_chi_square_cdf(y, nu, s);
  }

  template <typename T_y, typename T_dof, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_dof, T_scale>::type 
  cdf_function(const T_y& y, const T_dof& nu, const T_scale& s,
               const T3&, const T4&, const T5&) {
    using stan::math::gamma_q;
    using stan::math::gamma_q;

    return (gamma_q(nu * 0.5, 0.5 * nu * s * s / y));  
  }
};
