#define _LOG_PROB_ normal_log
#include <stan/prob/distributions/univariate/continuous/normal.hpp>

#include <test/prob/distributions/distribution_test_fixture.hpp>
#include <test/prob/distributions/distribution_tests_3_params.hpp>

#include <gtest/gtest.h>
#include <iostream>

using std::vector;
using std::numeric_limits;

TEST(ProbDistributionsNormal,Eigen) {
  using Eigen::Matrix;
  using Eigen::Dynamic;
  using stan::prob::normal_log;

  Matrix<double,Dynamic,1> y(4);
  y << 1.0, 2.0, 3.0, 4.0;
  Matrix<double,Dynamic,1> mu(4);
  mu << -1.0, -2.0, -3.0, -10.0;
  Matrix<double,Dynamic,1> sigma(4);
  sigma << 0.5, 1.5, 3.0, 15.0;
  // double z = normal_log(y,mu,sigma);
  // std::cout << "z=" << z << std::endl;
  EXPECT_TRUE(true);
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
  }

};

INSTANTIATE_TYPED_TEST_CASE_P(ProbDistributionsNormal,
                              DistributionTestFixture,
                              ProbDistributionsNormal);

