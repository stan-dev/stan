// Arguments: Ints, Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>
#include <boost/math/special_functions/binomial.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradCcdfLogBinomial : public AgradCcdfLogTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& ccdf_log) {
    vector<double> param(3);

    param[0] = 17;          // Successes
    param[1] = 45;          // Trials
    param[2] = 0.5;         // Probability
    parameters.push_back(param);
    ccdf_log.push_back(std::log(1.0 - 0.067578225422)); // expected ccdf_log
  }
  
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {

    // N (Trials)
    index.push_back(1U);
    value.push_back(-1);
      
    // p (Probability
    index.push_back(2U);
    value.push_back(-1e-4);

    index.push_back(2U);
    value.push_back(1+1e-4);
  }
  
  bool has_lower_bound() {
    return false;
  }
    
  bool has_upper_bound() {
    return false;
  }

  template <typename T_n, typename T_N, typename T_prob,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_prob>::type
  ccdf_log(const T_n& n, const T_N& N, const T_prob& theta,
          const T3&, const T4&, const T5&, const T6&, const T7&, 
          const T8&, const T9&) {
    return stan::prob::binomial_ccdf_log(n, N, theta);
  }


  template <typename T_n, typename T_N, typename T_prob,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_prob>::type
  ccdf_log_function(const T_n& n, const T_N& N, const T_prob& theta,
                   const T3&, const T4&, const T5&, const T6&, const T7&, 
                   const T8&, const T9&) {

    using std::log;
    using std::exp;
    using boost::math::binomial_coefficient;
      
    typename stan::return_type<T_prob>::type cdf(0);
 
    for (int i = 0; i <= n; i++) {
      cdf += binomial_coefficient<double>(N, i) * exp(i * log(theta) + (N - i) * log(1 - theta));
    }
      
    return log(1.0 - cdf);
      
  }
};
