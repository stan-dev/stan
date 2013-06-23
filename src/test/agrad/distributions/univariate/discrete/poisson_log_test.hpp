// Arguments: Ints, Doubles
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

#include <stan/math/functions/multiply_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsPoisson : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {

    using std::log;

    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = log(13.0);         // alpha
    parameters.push_back(param);
    log_prob.push_back(-2.900934); // expected log_prob

    param[0] = 192;          // n
    param[1] = log(42.0);         // alpha
    parameters.push_back(param);
    log_prob.push_back(-145.3547); // expected log_prob

    param[0] = 0;            // n
    param[1] = log(3.0);          // alpha
    parameters.push_back(param);
    log_prob.push_back(-3.0); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index,
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // alpha
    // all OK
  }

  template <class T_n, class T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_n, T_rate>::type 
  log_prob(const T_n& n, const T_rate& alpha, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::poisson_log_log(n, alpha);
  }

  template <bool propto, 
      class T_n, class T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  typename stan::return_type<T_n, T_rate>::type 
  log_prob(const T_n& n, const T_rate& alpha, const T2&,
     const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    return stan::prob::poisson_log_log<propto>(n, alpha);
  }
  
  
  template <class T_n, class T_rate, typename T2,
      typename T3, typename T4, typename T5, 
      typename T6, typename T7, typename T8, 
      typename T9>
  var log_prob_function(const T_n& n, const T_rate& alpha, const T2&,
      const T3&, const T4&, const T5&, const T6&, const T7&, const T8&, const T9&) {
    using std::exp;
    using boost::math::lgamma;
    using stan::math::multiply_log;
    using stan::prob::LOG_ZERO;
    using stan::prob::include_summand;

    var logp(0);

    if (alpha == -std::numeric_limits<double>::infinity())
      return n == 0 ? 0 : LOG_ZERO;
    
    if (alpha == std::numeric_limits<double>::infinity())
      return LOG_ZERO;
    
    if (include_summand<true>::value)
      logp -= lgamma(n + 1.0);
    if (include_summand<true,T_rate>::value)
      logp += n * alpha - exp(alpha);
    return logp;
  }
};
