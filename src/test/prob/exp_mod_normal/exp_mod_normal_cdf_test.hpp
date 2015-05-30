// Arguments: Doubles, Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/exp_mod_normal_cdf.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfExpModNormal : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& cdf) {
    vector<double> param(4);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //lambda
    parameters.push_back(param);
    cdf.push_back(0.2384217081348766198445);     // expected cdf

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 1; //lambda
    parameters.push_back(param);
    cdf.push_back(0.538079416212226213645); // expected cdf

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    param[3] = 3; //lambda
    parameters.push_back(param);
    cdf.push_back(0.0123402360230233393984); // expected cdf

    param[0] = -1.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 1.2;           // sigma
    param[3] = 1.9; //lambda
    parameters.push_back(param);
    cdf.push_back(0.0009426491340717019218332); // expected cdf
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
            typename T_inv_scale, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  cdf(const T_y& y, const T_loc& mu, const T_scale& sigma,
      const T_inv_scale& lambda, const T4&, const T5&) {
    return stan::math::exp_mod_normal_cdf(y, mu, sigma, lambda);
  }


  template <typename T_y, typename T_loc, typename T_scale,
            typename T_inv_scale, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale,T_inv_scale>::type 
  cdf_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
               const T_inv_scale& lambda, const T4&, const T5&) {

    return 0.5 * (1 + erf((y - mu) / (sqrt(2.0) * sigma))) - exp(-lambda * (y - mu) + lambda * sigma * lambda * sigma / 2.0) * (0.5 * (1 + erf(((y - mu) - sigma * lambda * sigma) / (sqrt(2.0) * sigma))));
  }
};
