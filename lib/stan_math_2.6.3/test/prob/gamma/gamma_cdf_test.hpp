// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/gamma_cdf.hpp>

#include <stan/math/prim/scal/fun/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCdfGamma : public AgradCdfTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 2.0;                 // alpha
    param[2] = 2.0;                 // beta
    parameters.push_back(param);
    cdf.push_back(0.5939941502901618930466); // expected cdf

    param[0] = 2.0;                 // y
    param[1] = 0.25;                // alpha
    param[2] = 0.75;                // beta
    parameters.push_back(param);
    cdf.push_back(0.9665835558410209582547);  // expected cdf

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    cdf.push_back(0.6321205588285576659757);       // expected cdf
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1.0);
    
    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // beta
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

  template <typename T_y, typename T_shape, typename T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_inv_scale>::type 
  cdf(const T_y& y, const T_shape& alpha, const T_inv_scale& beta,
      const T3&, const T4&, const T5&) {
    return stan::math::gamma_cdf(y, alpha, beta);
  }
  
  
  template <typename T_y, typename T_shape, typename T_inv_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_shape, T_inv_scale>::type 
  cdf_function(const T_y& y, const T_shape& alpha, const T_inv_scale& beta,
               const T3&, const T4&, const T5&) {
    using stan::math::gamma_p;
    using boost::math::gamma_p;

    return gamma_p(alpha, beta * y);
  }
};
