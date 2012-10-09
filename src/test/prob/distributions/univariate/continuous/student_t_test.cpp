#define _LOG_PROB_ student_t_log
#include <stan/prob/distributions/univariate/continuous/student_t.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_4_params.hpp>

using std::vector;
using std::numeric_limits;

class ProbDistributionsStudentT : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
		    vector<double>& log_prob) {
    vector<double> param(4);

    param[0] = 1.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.0;           // mu
    param[3] = 1.0;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.837877); // expected log_prob

    param[0] = -3.0;          // y
    param[1] = 2.0;           // nu
    param[2] = 0.0;           // mu
    param[3] = 1.0;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.596843); // expected log_prob

    param[0] = 2.0;           // y
    param[1] = 1.0;           // nu
    param[2] = 0.0;           // mu
    param[3] = 2.0;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.531024); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
		      vector<double>& value) {
    // y
    
    // nu
    index.push_back(1U);
    value.push_back(0.0);

    index.push_back(1U);
    value.push_back(-1.0);

    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // mu
    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(3U);
    value.push_back(0.0);

    index.push_back(3U);
    value.push_back(-1.0);

    index.push_back(3U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(3U);
    value.push_back(-numeric_limits<double>::infinity());
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsStudentT,
			      DistributionTestFixture,
			      ProbDistributionsStudentT);
