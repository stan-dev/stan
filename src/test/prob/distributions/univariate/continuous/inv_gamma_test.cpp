#define _LOG_PROB_ inv_gamma_log
#include <stan/prob/distributions/univariate/continuous/inv_gamma.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsInvGamma : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.0;                 // y
    param[1] = 1.0;                 // alpha
    param[2] = 1.0;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-1.0);       // expected log_prob

    param[0] = 0.5;                 // y
    param[1] = 2.9;                 // alpha
    param[2] = 3.1;                 // beta
    parameters.push_back(param);
    log_prob.push_back(-0.8185295); // expected log_prob
    
    /*
      param[0] = 0.0;                 // y
      param[1] = 2.9;                 // alpha
      param[2] = 3.1;                 // beta
      parameters.push_back(param);
      log_prob.push_back(log(0.0));   // expected log_prob
    */
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // alpha
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // beta
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsInvGamma,
			      DistributionTestFixture,
			      ProbDistributionsInvGamma);
