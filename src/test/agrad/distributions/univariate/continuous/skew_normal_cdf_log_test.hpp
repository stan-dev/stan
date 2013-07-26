// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/skew_normal.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfLogSkewNormal : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf_log) {
    vector<double> param(4);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //alpha
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.25));     // expected cdf_log

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //alpha
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.707861)); // expected cdf_log

    param[0] = -1;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 3; //alpha
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.0000562444337118770)); // expected cdf_log

    param[0] = -0.3;          // y
    param[1] = 0.1;           // mu
    param[2] = 1.2;           // sigma
    param[3] = 1.9; //alpha
    parameters.push_back(param);
    cdf_log.push_back(std::log(0.05529793)); // expected cdf_log
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

    //alpha
    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());

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
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_shape& alpha, const T4&, const T5&, const T6&, const T7&, 
          const T8&, const T9&) {
    return stan::prob::skew_normal_cdf_log(y, mu, sigma, alpha);
  }


  template <typename T_y, typename T_loc, typename T_scale,
      typename T_shape, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  cdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
               const T_shape& alpha, const T4&, const T5&, const T6&, 
               const T7&, const T8&, const T9&) {
    using stan::math::owens_t;
    return log(0.5 * erfc(-(y - mu) / (sqrt(2.0) * sigma)) - 2.0 
               * owens_t((y - mu) / sigma, alpha));
  }
};
