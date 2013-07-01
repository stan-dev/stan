// Arguments: Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/rayleigh.hpp>

#include <stan/math/constants.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCdfRayleigh : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(2);

    param[0] = 1;           // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.393469340287367);     // expected cdf

    param[0] = 2;           // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.864664716763387); // expected cdf

    param[0] = 3;          // y
    param[1] = 1;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.988891003461758); // expected cdf

    param[0] = 3.5;          // y
    param[1] = 7.2;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.111439); // expected cdf
  }
  
  void invalid_values(vector<size_t>& index, 
          vector<double>& value) {    
    // y
    index.push_back(0U);
    value.push_back(-1.0);

    // sigma
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
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

  template <typename T_y, typename T_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_scale>::type 
  cdf(const T_y& y, const T_scale& sigma, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::rayleigh_cdf(y, sigma);
  }


  template <typename T_y, typename T_scale, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
var  cdf_function(const T_y& y, const T_scale& sigma, const T2&,
         const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;

    var cdf(1.0);
    if (include_summand<true,T_y,T_scale>::value)
      cdf *= (1.0 - exp(-0.5 * y * y / (sigma * sigma)));
    return cdf;
  }
};
