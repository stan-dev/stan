#define _LOG_PROB_ binomial_logit_log
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_2_discrete_1_param.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBinomialLogit : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    using stan::math::logit;

    param[0] = 10;           // n
    param[1] = 20;           // N
    param[2] = logit(0.4);          // theta
    parameters.push_back(param);

    param[0] = 5;            // n
    param[1] = 15;           // N
    param[2] = logit(0.8);          // theta
    parameters.push_back(param);
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    
    // N
    index.push_back(1U);
    value.push_back(-1);
    
    // theta
    index.push_back(2U);
    value.push_back(std::numeric_limits<double>::infinity());
  }

  template <class T_n, class T_N, class T_prob,
            typename T3, typename T4, typename T5, 
            typename T6, typename T7, typename T8, 
            typename T9>
  var log_prob(const T_n& n, const T_N& N, const T_prob& alpha,
               const T3&, const T4&, const T5&,
               const T6&, const T7&, const T8&,
               const T9&) {
    
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::math::multiply_log;
    using stan::prob::include_summand;

    using stan::math::inv_logit;

    var logp(0);
    if (include_summand<true>::value)
      logp += binomial_coefficient_log(N,n);
    if (include_summand<true,T_prob>::value) 
      logp += multiply_log(n,inv_logit(alpha))
        + (N - n) * log1m(inv_logit(alpha));
    return logp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBinomialLogit,
                              AgradDistributionTestFixture,
                              AgradDistributionsBinomialLogit);
