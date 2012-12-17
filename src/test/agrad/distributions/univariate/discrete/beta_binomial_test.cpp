#define _LOG_PROB_ beta_binomial_log
#include <stan/prob/distributions/univariate/discrete/beta_binomial.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_2_discrete_2_params.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsBetaBinomial : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(4);

    param[0] = 10;           // n
    param[1] = 35;           // N
    param[2] = 2.5;          // alpha
    param[3] = 2.0;          // beta
    parameters.push_back(param);

    param[0] = 10;           // n
    param[1] = 35;           // N
    param[2] = 5.0;          // alpha
    param[3] = 3.0;          // beta
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

  template <class T_n, class T_N, class T_prob,
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

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBetaBinomial,
			      AgradDistributionTestFixture,
			      AgradDistributionsBetaBinomial);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBetaBinomial,
			      AgradDistributionTestFixture2,
			      AgradDistributionsBetaBinomial);
INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsBetaBinomial,
			      AgradDistributionTestFixture3,
			      AgradDistributionsBetaBinomial);
/*
template <typename T_size>
void expect_propto(int n1, int N1, T_size alpha1, T_size beta1,
                   int n2, int N2, T_size alpha2, T_size beta2,
                   std::string message) {
  expect_eq_diffs(stan::prob::beta_binomial_log<false>(n1,N1,alpha1,beta1),
                  stan::prob::beta_binomial_log<false>(n2,N2,alpha2,beta2),
                  stan::prob::beta_binomial_log<true>(n1,N1,alpha1,beta1),
                  stan::prob::beta_binomial_log<true>(n2,N2,alpha2,beta2),
                  message);
}

using stan::agrad::var;

TEST(AgradDistributionsBetaBinomial,Propto) {
  int n = 10;
  int N = 35;
  expect_propto<var>(n,N,2.5,2.0,
                     n,N,5.0,3.0,
                     "var: alpha and beta");
}
*/
