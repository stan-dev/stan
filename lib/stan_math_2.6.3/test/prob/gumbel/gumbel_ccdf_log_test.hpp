// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/gumbel_ccdf_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradCcdfLogGumbel : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);

    param[0] = 0.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.36787944117144232159));     // expected ccdf_log

    param[0] = 1.0;           // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.6922006275553463538654219)); // expected ccdf_log

    param[0] = -2.0;          // y
    param[1] = 0.0;           // mu
    param[2] = 1.0;           // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.000617978989331093498619521)); // expected ccdf_log

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.1203922620798295861862650)); // expected ccdf_log
  }
  
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // mu
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
  ccdf_log(const T_y& y, const T_loc& mu, const T_scale& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::gumbel_ccdf_log(y, mu, beta);
  }


  template <typename T_y, typename T_loc, typename T_scale,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_loc, T_scale>::type 
  ccdf_log_function(const T_y& y, const T_loc& mu, const T_scale& beta,
                    const T3&, const T4&, const T5&) {
    return log(1.0 - exp(-exp(-(y - mu) / beta)));
  }
};
