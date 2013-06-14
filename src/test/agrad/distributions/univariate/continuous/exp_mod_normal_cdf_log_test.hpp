// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/exp_mod_normal.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfLogExpModNormal : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf_log) {
    vector<double> param(4);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //lambda
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.2384216994));     // expected cdf_log

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //lambda
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.5380794103)); // expected cdf_log
  }
  
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(0.0);

    //lambda
    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(0.0);

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());
  }
  
  bool has_lower_bound() {
    return false;
  }
  
  bool has_upper_bound() {
    return false;
  }

  template <typename T_y, typename T_loc, typename T_scale,
      typename T_inv_scale, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::exp_mod_normal_cdf_log(y, mu, sigma, lambda);
  }


  template <typename T_y, typename T_loc, typename T_scale,
      typename T_inv_scale, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  cdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
         const T_inv_scale& lambda, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {

    return log(0.5 * (1 + erf((y - mu) / (sqrt(2.0) * sigma))) - exp(-lambda * (y - mu) + lambda * sigma * lambda * sigma / 2.0) * 0.5 * (1 + erf(((y - mu) - sigma * lambda * sigma) / (sqrt(2.0) * sigma))));
  }
};
