// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/lognormal.hpp>

#include <stan/math/functions/square.hpp>
#include <stan/math/constants.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfLognormal : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 1.5;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.4687341); // expected cdf

    param[0] = 12.0;          // y
    param[1] = 3.0;           // mu
    param[2] = 0.9;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.2835506); // expected cdf
  }
 
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    index.push_back(0U);
    value.push_back(-numeric_limits<double>::infinity());
   
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
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
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
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::lognormal_cdf(y, mu, sigma);
  }
  
  template <typename T_y, typename T_loc, typename T_scale,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return 0.5 * erfc(-(log(y) - mu) / (std::sqrt(2) * sigma));
  }
};
