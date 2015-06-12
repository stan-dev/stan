// Arguments: Doubles, Doubles, Doubles
#include <stan/math/prim/scal/prob/uniform_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsUniform : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0.1;                // y
    param[1] = -0.1;               // alpha
    param[2] = 0.8;                // beta
    parameters.push_back(param);
    log_prob.push_back(log(1/0.9));   // expected log_prob

    param[0] = 0.2;                // y
    param[1] = -0.25;              // alpha
    param[2] = 0.25;               // beta
    parameters.push_back(param);
    log_prob.push_back(log(2.0));   // expected log_prob

    param[0] = 0.05;               // y
    param[1] = -5;                 // alpha
    param[2] = 5;                  // beta
    parameters.push_back(param);
    log_prob.push_back(log(0.1));   // expected log_prob
  }
 
  void invalid_values(vector<size_t>& /*index*/, 
                      vector<double>& /*value*/) {
    // y
    
    // alpha

    // beta
  }

  template <class T_y, class T_low, class T_high,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_low, T_high>::type 
  log_prob(const T_y& y, const T_low& alpha, const T_high& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::uniform_log(y, alpha, beta);
  }

  template <bool propto, 
            class T_y, class T_low, class T_high,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_low, T_high>::type 
  log_prob(const T_y& y, const T_low& alpha, const T_high& beta,
           const T3&, const T4&, const T5&) {
    return stan::math::uniform_log<propto>(y, alpha, beta);
  }
  
  
  template <class T_y, class T_low, class T_high,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_y, T_low, T_high>::type 
  log_prob_function(const T_y& y, const T_low& alpha, const T_high& beta,
                    const T3&, const T4&, const T5&) {
      using stan::math::LOG_ZERO;

      if (y < alpha || y > beta)
        return LOG_ZERO;

      return -log(beta - alpha);
  }
};
