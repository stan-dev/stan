#define _LOG_PROB_ bernoulli_log
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/discrete_distribution_tests_2_params.hpp>

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

  template <class T_prob>
  var log_prob(const int n, const T_prob& theta) {

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



#include <gtest/gtest.h>
#include <test/agrad/distributions/expect_eq_diffs.hpp>
#include <stan/agrad/agrad.hpp>
#include <stan/agrad/special_functions.hpp>
#include <stan/meta/traits.hpp>
#include <stan/prob/distributions/univariate/discrete/bernoulli.hpp>

template <typename T_prob>
void expect_propto(unsigned int n1, T_prob alpha1, 
                   unsigned int n2, T_prob alpha2, 
                   std::string message) {
  expect_eq_diffs(stan::prob::bernoulli_log<false>(n1, alpha1),
                  stan::prob::bernoulli_log<false>(n2, alpha2),
                  stan::prob::bernoulli_log<true>(n1, alpha1),
                  stan::prob::bernoulli_log<true>(n2, alpha2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBernoulli,Propto) {
  unsigned int n;
  n = 0;
  expect_propto<var>(n, 0.3,
                     n, 0.75,
                     "var: prob, n=0");
  n = 1;
  expect_propto<var>(n, 0.3,
                     n, 0.75,
                     "var: prob, n=1");

}
