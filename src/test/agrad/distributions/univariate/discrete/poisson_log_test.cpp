#define _LOG_PROB_ poisson_log_log
#include <stan/prob/distributions/univariate/discrete/poisson.hpp>

#include <test/agrad/distributions/distribution_test_fixture.hpp>
#include <test/agrad/distributions/distribution_tests_1_discrete_1_param.hpp>

using std::vector;
using std::numeric_limits;
using stan::agrad::var;

class AgradDistributionsPoissonLog : public AgradDistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters) {
    using std::log;

    vector<double> param(2);

    param[0] = 17;           // n
    param[1] = log(13.0);         // lambda
    parameters.push_back(param);

    param[0] = 192;          // n
    param[1] = log(42.0);         // lambda
    parameters.push_back(param);

    param[0] = 0;            // n
    param[1] = log(3.0);          // lambda
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

  }

  template <class T_log_rate>
  var log_prob(const int n, const T_log_rate& alpha) {

    using std::exp;
    using boost::math::lgamma;
    using stan::math::multiply_log;
    using stan::prob::LOG_ZERO;
    using stan::prob::include_summand;

    var logp(0);

    if (log(alpha) == 0)
      return n == 0 ? 0 : LOG_ZERO;
    
    if (std::isinf(log(alpha)))
      return LOG_ZERO;
    
    if (include_summand<true>::value)
      logp -= lgamma(n + 1.0);
    if (include_summand<true,T_log_rate>::value)
      logp += n * alpha - exp(alpha);
    return logp;
  }
};

INSTANTIATE_TYPED_TEST_CASE_P(AgradDistributionsPoissonLog,
                              AgradDistributionTestFixture,
                              AgradDistributionsPoissonLog);

