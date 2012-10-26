#define _LOG_PROB_ bernoulli_logit_log
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBernoulliLogistic : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    using stan::math::logit;
    using std::exp;
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = logit(0.25); // theta
    parameters.push_back(param);

    param[0] = 0;           // n
    param[1] = logit(0.25); // theta
    parameters.push_back(param);

    param[0] = 1;           // n
    param[1] = logit(0.01); // theta
    parameters.push_back(param);

    param[0] = 0;           // n
    param[1] = logit(0.01); // theta
    parameters.push_back(param);

    param[0] = 0;            // n
    param[1] = 25;           // theta
    parameters.push_back(param);

    param[0] = 1;            // n
    param[1] = -25;          // theta
    parameters.push_back(param);
    
    param[0] = 0;           // n
    param[1] = -25;         // theta
    parameters.push_back(param);
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    index.push_back(0U);
    value.push_back(-1);

    index.push_back(0U);
    value.push_back(2);

    // theta
  }

  template <class T_prob>
  var log_prob(const int n, const T_prob& theta) {

    using std::log;
    using stan::math::log1m;
    using stan::prob::include_summand;

    if (include_summand<true,T_prob>::value) {
      T_prob ntheta = (2*n-1) * theta;
      // Handle extreme values gracefully using Taylor approximations.
      const static double cutoff = 20.0;
      if (ntheta > cutoff)
	return -exp(-ntheta);
      else if (ntheta < -cutoff)
	return ntheta;
      else
	return -log(1 + exp(-ntheta));
    }
    return 0.0;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBernoulliLogistic,
			      AgradDistributionTestFixture,
			      AgradDistributionsBernoulliLogistic);

