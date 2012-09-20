#define _LOG_PROB_ normal_log
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>
#include <gtest/gtest.h>

#include <boost/random/mersenne_twister.hpp>

#include <iostream>

using std::vector;
using std::numeric_limits;

TEST(ProbDistributionsNormal, random) {
  boost::random::mt19937 rng;
  double variate = stan::prob::normal_random(2.0,1.0,rng);
  std::cout << "variate=" << variate << std::endl;
  EXPECT_EQ(1,1);
}

class ProbDistributionsNormal : public DistributionTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& log_prob) {
    vector<double> param(3);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-0.9189385); // expected log_prob

    param[0] = 1;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-1.418939); // expected log_prob

    param[0] = -2;          // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-2.918939); // expected log_prob

    param[0] = -3.5;          // y
    param[1] = 1.9;           // mu
    param[2] = 7.2;           // sigma
    parameters.push_back(param);
    log_prob.push_back(-3.174270); // expected log_prob
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
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsNormal,
                              DistributionTestFixture,
                              ProbDistributionsNormal);


TEST(ProbDistributions,NormalCdf) {
  using stan::prob::normal_cdf;
  EXPECT_NEAR(0.5, normal_cdf(0,0,0.01), 1E-8);
  EXPECT_NEAR(0.5, normal_cdf(0,0,0.1), 1E-8);
  EXPECT_NEAR(0.5, normal_cdf(0,0,1), 1E-8);
  EXPECT_NEAR(0.5, normal_cdf(0,0,10), 1E-8);
  EXPECT_NEAR(0.5, normal_cdf(0,0,100), 1E-8);

  EXPECT_NEAR(0.3694413, normal_cdf(1,5,12), 1E-7);
  EXPECT_NEAR(0.9999683, normal_cdf(-2,-3,0.25), 1E-7);
}
