#define _LOG_PROB_ normal_log
#define _CDF_ normal_cdf
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


class ProbCumulativeNormal : public CumulativeTest {
public:
  void valid_values(vector<vector<double> >& parameters,
                    vector<double>& cdf) {
    vector<double> param(3);

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 0.01;        // sigma
    parameters.push_back(param);
    cdf.push_back(0.5);     // expected cdf

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 0.1;         // sigma
    parameters.push_back(param);
    cdf.push_back(0.5);     // expected cdf

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.5);     // expected cdf

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 1;           // sigma
    parameters.push_back(param);
    cdf.push_back(0.5);     // expected cdf

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 10;          // sigma
    parameters.push_back(param);
    cdf.push_back(0.5);     // expected cdf

    param[0] = 0;           // y
    param[1] = 0;           // mu
    param[2] = 100;         // sigma
    parameters.push_back(param);
    cdf.push_back(0.5);     // expected cdf

    param[0] = 1;           // y
    param[1] = 5;           // mu
    param[2] = 12;          // sigma
    parameters.push_back(param);
    cdf.push_back(0.3694413); // expected cdf

    param[0] = -2;          // y
    param[1] = -3;          // mu
    param[2] = 0.25;        // sigma
    parameters.push_back(param);
    cdf.push_back(0.9999683); // expected cdf
  }

  void zero_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = -std::numeric_limits<double>::infinity(); // y
    param[1] = 0;           // mu
    param[2] = 0.01;        // sigma
    parameters.push_back(param);
  }

  void one_values(vector<vector<double> >& parameters) {
    vector<double> param(3);

    param[0] = std::numeric_limits<double>::infinity(); // y
    param[1] = 0;           // mu
    param[2] = 0.01;        // sigma
    parameters.push_back(param);
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

INSTANTIATE_TYPED_TEST_CASE_P(ProbCumulativeNormal,
                              CumulativeTestFixture,
                              ProbCumulativeNormal);

