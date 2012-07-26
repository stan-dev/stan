#ifndef __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TEST_FIXTURE_HPP___
#define __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TEST_FIXTURE_HPP___

#ifndef _LOG_PROB_
#define _LOG_PROB_ MUST_DEFINE_LOG_PROB
#endif

using stan::prob::_LOG_PROB_;

#include <gtest/gtest.h>
#include <stdexcept>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix.hpp>

using std::vector;

using boost::math::policies::policy;
using boost::math::policies::domain_error;
using boost::math::policies::errno_on_error;

using stan::math::default_policy;

typedef policy<
  domain_error<errno_on_error>
  > errno_policy;

class DistributionTest {
public:
  /**
   * Subclasses should define this function.
   *
   */
  virtual void valid_values(vector<vector<double> >& parameters, 
			    vector<double>& log_prob) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& index, 
			      vector<double>& value) {
    throw std::runtime_error("valid_values() not implemented");
  }

};

template<class T>
class DistributionTestFixture : public ::testing::Test {
public:
  vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> expected_log_prob;
    T().valid_values(params, expected_log_prob); 
    return params[0];
  }

  double first_valid_value() {
    vector<vector<double> > params;
    vector<double> expected_log_prob;
    T().valid_values(params, expected_log_prob); 
    return expected_log_prob[0];
  }

};

TYPED_TEST_CASE_P(DistributionTestFixture);

#endif
