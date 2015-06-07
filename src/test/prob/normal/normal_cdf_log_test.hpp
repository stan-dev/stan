// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/normal_cdf_log.hpp>

#include <stan/math/prim/scal/fun/constants.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfLogNormal : public AgradCdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf_log) {
    vector<double> param(3);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    cdf_log.push_back(-0.6931471805599452862268);     // expected cdf_log

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    cdf_log.push_back(-0.1727537790234499326392); // expected cdf_log

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    cdf_log.push_back(-3.78318433368203166367); // expected cdf_log

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    parameters.push_back(param);
    cdf_log.push_back(-1.484448229919656192521); // expected cdf_log
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }
  
  bool has_lower_bound() {
    return false;
  }
  
  bool has_upper_bound() {
    return false;
  }

  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  cdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
          const T3&, const T4&, const T5&) {
    return stan::math::normal_cdf_log(y, mu, sigma);
  }


  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  cdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                   const T3&, const T4&, const T5&) {
    using stan::math::SQRT_2;
    using std::log;
    return log(0.5 + 0.5 * erf((y - mu) / (sigma * SQRT_2)));
  }
};
