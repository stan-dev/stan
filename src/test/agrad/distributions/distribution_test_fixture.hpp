#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TEST_FIXTURE_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TEST_FIXTURE_HPP___

#ifndef _LOG_PROB_
#define _LOG_PROB_ MUST_DEFINE_LOG_PROB
#endif

using stan::prob::_LOG_PROB_;

#include <gtest/gtest.h>
#include <stdexcept>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix.hpp>

using std::vector;
using Eigen::Matrix;
using Eigen::Dynamic;

using boost::math::policies::policy;
using boost::math::policies::domain_error;
using boost::math::policies::errno_on_error;

using stan::math::default_policy;

typedef policy<
  domain_error<errno_on_error>
  > errno_policy;

class AgradDistributionTest {
public:
  /**
   * Subclasses should define this function.
   *
   */
  virtual void valid_values(vector<vector<double> >& parameters) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& index, 
			      vector<double>& value) {
    throw std::runtime_error("valid_values() not implemented");
  }
};

template<class T>
class AgradDistributionTestFixture : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  
  double e() {
    return 1e-8;
  }
};
template<class T>
class AgradDistributionTestFixture2 : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  double e() {
    return 1e-8;
  }
};
template<class T>
class AgradDistributionTestFixture3 : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  double e() {
    return 1e-8;
  }
};
template<class T>
class AgradDistributionTestFixture4 : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  double e() {
    return 1e-8;
  }
};
template<class T>
class AgradDistributionTestFixture5 : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  double e() {
    return 1e-8;
  }
};
template<class T>
class AgradDistributionTestFixture6 : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  double e() {
    return 1e-8;
  }
};
template<class T>
class AgradDistributionTestFixture7 : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    T().valid_values(params); 
    return params[0];
  }
  double e() {
    return 1e-8;
  }
};
TYPED_TEST_CASE_P(AgradDistributionTestFixture);
TYPED_TEST_CASE_P(AgradDistributionTestFixture2);
TYPED_TEST_CASE_P(AgradDistributionTestFixture3);
TYPED_TEST_CASE_P(AgradDistributionTestFixture4);
TYPED_TEST_CASE_P(AgradDistributionTestFixture5);
TYPED_TEST_CASE_P(AgradDistributionTestFixture6);
TYPED_TEST_CASE_P(AgradDistributionTestFixture7);
#endif
