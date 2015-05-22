// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/double_exponential_ccdf_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCcdfLogDoubleExponential : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);
    
    param[0] = 1.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.5));  // expected ccdf_log

    param[0] = 2.0;                  // y
    param[1] = 1.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.8160602794142788392022381149192695662770944344841160827460815991512692521275500983214263628270401781));   // expected ccdf_log
    
    param[0] = -3.0;                 // y
    param[1] = 2.0;                  // mu
    param[2] = 1.0;                  // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.003368973499542733548318024211574212124424792513677542715152765786341761257802031140724569422104180774));   // expected ccdf_log
    
    param[0] = 1.0;                  // y
    param[1] = 0.0;                  // mu
    param[2] = 2.0;                  // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.6967346701436832881981002325044097732790409322564065221585539206324717402931257880006761942460052720));   // expected ccdf_log

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.5;                  // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.224664482058610795715051192507781));        // expected ccdf_log

    param[0] = 1.9;                  // y
    param[1] = 2.3;                  // mu
    param[2] = 0.25;                  // sigma
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.10094825899732770424258963382167488));       // expected ccdf_log
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
    value.push_back(numeric_limits<double>::infinity());

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
  ccdf_log(const T_y& y, const T_loc& mu, const T_scale& sigma,
           const T3&, const T4&, const T5&) {
    return stan::math::double_exponential_ccdf_log(y, mu, sigma);
  }
  
  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  ccdf_log_function(const T_y& y, const T_loc& mu, const T_scale& sigma,
                    const T3&, const T4&, const T5&) {
    return stan::math::double_exponential_ccdf_log(y, mu, sigma);
  }
};

