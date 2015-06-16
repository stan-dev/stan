// Arguments: Ints, Ints, Doubles
#include <stan/math/prim/scal/prob/binomial_logit_log.hpp>

#include <stan/math/prim/scal/fun/logit.hpp>
#include <stan/math/prim/scal/fun/binomial_coefficient_log.hpp>
#include <stan/math/prim/scal/fun/log1m.hpp>
#include <stan/math/prim/scal/fun/multiply_log.hpp>
#include <stan/math/prim/scal/fun/inv_logit.hpp>

using std::vector;
using std::numeric_limits;
using stan::math::var;

class AgradDistributionsBinomialLogit : public AgradDistributionTest {
public:
    void valid_values(vector<vector<double> >& parameters,
                      vector<double>& log_prob) {
    vector<double> param(3);
    using stan::math::logit;

    param[0] = 10;           // n
    param[1] = 20;           // N
    param[2] = logit(0.4);          // alpha
    parameters.push_back(param);
    log_prob.push_back(-2.144372241799002765106); // expected log_prob

    param[0] = 5;            // n
    param[1] = 15;           // N
    param[2] = logit(0.8);          // alpha
    parameters.push_back(param);
    log_prob.push_back(-9.202729812928724939525); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    // N
    index.push_back(1U);
    value.push_back(-1);
    
    // alpha
    index.push_back(2U);
    value.push_back(std::numeric_limits<double>::infinity());
  }

  template <class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type 
  log_prob(const T_n& n, const T_N& N, const T_prob& alpha,
           const T3&, const T4&, const T5&) {
    return stan::math::binomial_logit_log(n, N, alpha);
  }

  template <bool propto, 
            class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type 
  log_prob(const T_n& n, const T_N& N, const T_prob& alpha,
           const T3&, const T4&, const T5&) {
    return stan::math::binomial_logit_log<propto>(n, N, alpha);
  }
  
  
  template <class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5>
  typename stan::return_type<T_prob>::type 
  log_prob_function(const T_n& n, const T_N& N, const T_prob& alpha,
                        const T3&, const T4&, const T5&) {
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::math::multiply_log;
    using stan::math::inv_logit;

    return binomial_coefficient_log(N,n) + multiply_log(n,inv_logit(alpha)) + (N - n) * log1m(inv_logit(alpha));
  }
};
