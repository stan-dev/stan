#define _LOG_PROB_ binomial_log
#include <stan/prob/distributions/univariate/discrete/binomial.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_2_discrete_1_param.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 20;           // N
    param[2] = 0.4;          // theta
    parameters.push_back(param);

    param[0] = 5;            // n
    param[1] = 15;           // N
    param[2] = 0.8;          // theta
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
    value.push_back(-1e-15);
    
    index.push_back(2U);
    value.push_back(1.0+1e-15);
  }

  template <class T_n=int, class T_N=int, class T_prob,
	    typename T3, typename T4, typename T5, 
	    typename T6, typename T7, typename T8, 
	    typename T9>
  var log_prob(const T_n& n, const T_N& N, const T_prob& theta,
	       const T3&, const T4&, const T5&,
	       const T6&, const T7&, const T8&,
	       const T9&) {
    
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::math::multiply_log;
    using stan::prob::include_summand;

    var logp(0);
    if (include_summand<true>::value)
      logp += binomial_coefficient_log(N,n);
    if (include_summand<true,T_prob>::value) 
      logp += multiply_log(n,theta)
	+ (N - n) * log1m(theta);
    return logp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBinomial,
			      AgradDistributionTestFixture,
			      AgradDistributionsBinomial);
