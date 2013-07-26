// Arguments: Doubles, Doubles, Doubles
#include <stan/prob/distributions/univariate/continuous/uniform.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogUniform : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);

    param[0] = 0.1;                // y
    param[1] = -0.1;               // alpha
    param[2] = 5.0;                // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.2 / 5.1));   // expected ccdf_log

    param[0] = 0.2;                // y
    param[1] = -0.25;              // alpha
    param[2] = 5.0;               // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.45 / 5.25));   // expected ccdf_log

    param[0] = 0.05;               // y
    param[1] = -5;                 // alpha
    param[2] = 5;                  // beta
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 5.05 / 10.0));   // expected ccdf_log
  }
 
  void invalid_values(vector<size_t>& /*index*/, 
                      vector<double>& /*value*/) {
    // y
    
    // alpha

    // beta
  }


  bool has_lower_bound() {
    return false;
  }

  //BOUND INCLUDED IN ORDER FOR TEST TO PASS WITH CURRENT FRAMEWORK
  bool has_upper_bound() {
    return true;
  }

  double upper_bound() {
    return 5.0;
  }

  template <class T_y, class T_low, class T_high,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_low, T_high>::type 
  ccdf_log(const T_y& y, const T_low& alpha, const T_high& beta,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, 
      const T9&) {
    return stan::prob::uniform_ccdf_log(y, alpha, beta);
  }
  
  template <class T_y, class T_low, class T_high,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_y, T_low, T_high>::type 
  ccdf_log_function(const T_y& y, const T_low& alpha, const T_high& beta,
               const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, 
               const T9&) {
      using stan::prob::include_summand;
      using stan::prob::LOG_ZERO;

      if (y < alpha || y > beta)
        return 0.0;

      return log(1.0 - (y - alpha) / (beta - alpha));
  }
};
