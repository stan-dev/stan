#define _LOG_PROB_ logistic_log
#include <stan/prob/distributions/univariate/continuous/logistic.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>
#include <gtest/gtest.h>

#include <iostream>

using std::vector;
using std::numeric_limits;

class ProbDistributionsLogistic : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 1.2;           // y
    param[1] = 0.3;           // mu
    param[2] = 2.0;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.129645); // expected log_prob

    param[0] = -1.0;          // y
    param[1] = 0.2;           // mu
    param[2] = 0.25;          // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.430098); // expected log_prob
  }
 
  void invalid_values(vector<size_t>& index, 
                      vector<double>& value) {
    // y
    
    // mu
    index.push_back(1U);
    value.push_back(numeric_limits<double>::infinity());

    index.push_back(1U);
    value.push_back(-numeric_limits<double>::infinity());

    // sigma
    index.push_back(2U);
    value.push_back(0.0);

    index.push_back(2U);
    value.push_back(-1.0);

    index.push_back(2U);
    value.push_back(-numeric_limits<double>::infinity());

    index.push_back(2U);
    value.push_back(numeric_limits<double>::infinity());
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsLogistic,
                              DistributionTestFixture,
                              ProbDistributionsLogistic);

TEST(ProbDistributionsLogisticCDF, Values) {
    EXPECT_FLOAT_EQ(0.047191944, stan::prob::logistic_cdf(-3.45, 5.235, 2.89));
}