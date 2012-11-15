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

using stan::math::default_policy;

typedef boost::math::policies::policy<
  boost::math::policies::domain_error<boost::math::policies::errno_on_error>
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

/**
 * Utility functions
 **/
using stan::agrad::var;
using stan::scalar_type;
using stan::is_vector;
using stan::is_constant;
using stan::is_constant_struct;

template<class T>
T get_params(vector<vector<double> >& parameters, size_t p) {
  return parameters[0][p];
}
template<>
vector<double> get_params<vector<double> >(vector<vector<double> >& parameters, size_t p) {
  vector<double> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    param[n] = parameters[n][p];
  return param;
}
template<>
vector<var> get_params<vector<var> >(vector<vector<double> >& parameters, size_t p) {
  vector<var> param(parameters.size());
  for (size_t n = 0; n < parameters.size(); n++)
    param[n] = parameters[n][p];
  return param;
}
template<class T>
double get_param(vector<vector<double> >& parameters, size_t n, size_t p) {
  if (is_vector<T>::value)
    return parameters[n][p];
  else
    return parameters[0][p];
}



#endif
