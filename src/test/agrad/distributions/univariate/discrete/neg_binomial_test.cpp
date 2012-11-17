#define _LOG_PROB_ neg_binomial_log
#include <stan/prob/distributions/univariate/discrete/neg_binomial.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_1_discrete_2_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsNegBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = 10;           // n
    param[1] = 2.0;          // alpha
    param[2] = 1.5;          // beta
    parameters.push_back(param);

    param[0] = 100;          // n
    param[1] = 3.0;          // alpha
    param[2] = 3.5;          // beta
    parameters.push_back(param);
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);
    
    // alpha
    index.push_back(1U);
    value.push_back(0);
    
    // beta
    index.push_back(2U);
    value.push_back(0);
  }

  template <class T_shape, class T_inv_scale>
  var log_prob(const int n, const T_shape& alpha, const T_inv_scale& beta) {
    
    using std::log;
    using stan::math::binomial_coefficient_log;
    using stan::math::log1m;
    using stan::prob::include_summand;

    var logp(0);
    // Special case where negative binomial reduces to Poisson
    if (alpha > 1e10) {
      if (include_summand<true>::value)
	logp -= lgamma(n + 1.0);
      if (include_summand<true,T_shape>::value ||
	  include_summand<true,T_inv_scale>::value) {
	typename stan::return_type<T_shape, T_inv_scale>::type lambda;
	lambda = alpha / beta;
	logp += multiply_log(n, lambda) - lambda;
      }
      return logp;
    }
    // More typical cases
    if (include_summand<true,T_shape>::value)
      if (n != 0)
	logp += binomial_coefficient_log<typename stan::scalar_type<T_shape>::type>
	  (n + alpha - 1.0, n);
    if (include_summand<true,T_shape,T_inv_scale>::value)
      logp += -n * log1p(beta) 
	+ alpha * log(beta / (1 + beta));
    return logp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsNegBinomial,
			      AgradDistributionTestFixture,
			      AgradDistributionsNegBinomial);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsNegBinomial,
			      AgradDistributionTestFixture2,
			      AgradDistributionsNegBinomial);
