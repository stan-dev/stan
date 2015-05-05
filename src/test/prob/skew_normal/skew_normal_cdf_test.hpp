// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/skew_normal_cdf.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfSkewNormal : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //alpha
    parameters.push_back(param);
    cdf.push_back(0.2500000000000001110223);     // expected cdf

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //alpha
    parameters.push_back(param);
    cdf.push_back(0.7078609817371410706244); // expected cdf

    param[0] = -1;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 3; //alpha
    parameters.push_back(param);
    cdf.push_back(0.00005624443371187709415635692284235862134252368571012732206706451807484923689534924660749487716440408152); // expected cdf

    param[0] = -0.3;          // y
    param[1] = 0.1;           // mu
    param[2] = 1.2;           // sigma
    param[3] = 1.9; //alpha
    parameters.push_back(param);
    cdf.push_back(0.05529792943083011724781); // expected cdf
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
            typename T_shape, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_shape& alpha, const T4&, const T5&) {
    return stan::math::skew_normal_cdf(y, mu, sigma, alpha);
  }


  template <typename T_y, typename T_loc, typename T_scale,
            typename T_shape, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale,T_shape>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
               const T_shape& alpha, const T4&, const T5&) {
    using stan::math::owens_t;
    return 0.5 * erfc(-(y - mu) / (sqrt(2.0) * sigma)) - 2.0 * owens_t((y - mu) / sigma, alpha);
  }
};
