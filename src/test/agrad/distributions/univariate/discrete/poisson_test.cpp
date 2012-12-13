#define _LOG_PROB_ poisson_log
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsPoisson : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = 13.0;         // lambda
    parameters.push_back(param);

    param[0] = 192;          // n
    param[1] = 42.0;         // lambda
    parameters.push_back(param);

    param[0] = 0;            // n
    param[1] = 3.0;          // lambda
    parameters.push_back(param);

    /*param[0] = 0;            // n
    param[1] = std::numeric_limits<double>::infinity(); // lambda
    parameters.push_back(param);*/

    /*    param[0] = 1;            // n
    param[1] = 0.0;          // lambda
    parameters.push_back(param);*/
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // n
    index.push_back(0U);
    value.push_back(-1);

    // lambda
    index.push_back(1U);
    value.push_back(-1e-5);

    index.push_back(1U);
    value.push_back(-1);
  }

  template <class T_n, class T_rate,
	    typename T2, typename T3, typename T4, 
	    typename T5, typename T6, typename T7, 
	    typename T8, typename T9>
  var log_prob(const T_n& n, const T_rate& lambda,
	       const T2&, const T3&, const T4&,
	       const T5&, const T6&, const T7&,
	       const T8&, const T9&) {

    using boost::math::lgamma;
    using stan::math::multiply_log;
    using stan::prob::LOG_ZERO;
    using stan::prob::include_summand;

    var logp(0);

    if (lambda == 0)
      return n == 0 ? 0 : LOG_ZERO;
    
    if (std::isinf(lambda))
      return LOG_ZERO;
    
    if (include_summand<true>::value)
      logp -= lgamma(n + 1.0);
    if (include_summand<true,T_rate>::value)
      logp += multiply_log(n, lambda) - lambda;
    return logp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsPoisson,
			      AgradDistributionTestFixture,
			      AgradDistributionsPoisson);

