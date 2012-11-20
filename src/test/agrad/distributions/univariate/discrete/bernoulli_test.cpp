#define _LOG_PROB_ bernoulli_log
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBernoulli : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(2);

    param[0] = 1;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);

    param[0] = 0;           // n
    param[1] = 0.25;        // theta
    parameters.push_back(param);

    param[0] = 1;           // n
    param[1] = 0.01;        // theta
    parameters.push_back(param);

    param[0] = 0;           // n
    param[1] = 0.01;        // theta
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
    index.push_back(1U);
    value.push_back(-0.001);

    index.push_back(1U);
    value.push_back(1.001);
  }

  template <class T_n=int, class T_prob,
	    typename T2, typename T3, typename T4,
	    typename T5, typename T6, typename T7,
	    typename T8, typename T9>
  var log_prob(const T_n& n, const T_prob& theta,
	       const T2&, const T3&, const T4&, 
	       const T5&, const T6&, const T7&, 
	       const T8&, const T9&) {
    using std::log;
    using stan::math::log1m;
    using stan::prob::include_summand;

    var logp(0);
    if (include_summand<true,T_prob>::value) {
      if (n == 1)
	logp += log(theta);
      else if (n == 0)
	logp += log1m(theta);
    }
    return logp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBernoulli,
			      AgradDistributionTestFixture,
			      AgradDistributionsBernoulli);


