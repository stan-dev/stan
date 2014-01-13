// Arguments: Ints, Ints, Ints, Ints
#include <stan/prob/distributions/univariate/discrete/hypergeometric.hpp>

#include <stan/math/functions/binomial_coefficient_log.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsNegBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
        vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 5;           // n
    param[1] = 15;          // N
    param[2] = 10;          // a
    param[3] = 10;          // b
    parameters.push_back(param);
    log_prob.push_back(-4.119424); // expected log_prob

    param[0] = 5;           // n
    param[1] = 15;          // N
    param[2] = 10;          // a
    param[3] = 10;          // b
    parameters.push_back(param);
    log_prob.push_back(-4.119424); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    // N
    index.push_back(1U);
    value.push_back(-1);

    // a
    index.push_back(2U);
    value.push_back(-1);

    // b
    index.push_back(3U);
    value.push_back(-1);
  }

  template <class T_n, class T_N, class T_a, class T_b,
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  typename stan::return_type<T_n,T_N,T_a,T_b>::type 
  log_prob(const T_n& n, const T_N& N, const T_a& a, const T_b& b,
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::hypergeometric_log(n, N, a, b);
  }

  template <bool propto, 
      class T_n, class T_N, class T_a, class T_b,
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  double
  log_prob(const T_n& n, const T_N& N, const T_a& a, const T_b& b,
     const T4&, const T5&, const T6&, 
     const T7&, const T8&, const T9&) {
    return stan::prob::hypergeometric_log<propto>(n, N, a, b);
  }
  

  template <class T_n, class T_N, class T_a, class T_b,
      typename T4, typename T5, typename T6, 
      typename T7, typename T8, typename T9>
  var log_prob_function(const T_n& n, const T_N& N, const T_a& a, const T_b& b,
      const T4&, const T5&, const T6&, 
      const T7&, const T8&, const T9&) {
    using stan::prob::include_summand;
    using stan::math::binomial_coefficient_log;
    
    var logp(0);
    logp += binomial_coefficient_log(a, n)
      + binomial_coefficient_log(b, N-n)
      - binomial_coefficient_log(a+b, N);
    return logp;
  }
};
