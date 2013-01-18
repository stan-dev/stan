#ifndef __TEST__AGRAD__DISTRIBUTIONS__NEW_DISTRIBUTION_TEST_FIXTURE_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__NEW_DISTRIBUTION_TEST_FIXTURE_HPP___

#include <stdexcept>
#include <stan/math/error_handling.hpp>
#include <stan/math/matrix.hpp>

struct empty {};

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
  virtual void valid_values(vector<vector<double> >& /*parameters*/,
			    vector<double>& /* log_prob */) {
    throw std::runtime_error("valid_values() not implemented");
  }
  
  // don't need to list nan. checked by the test.
  virtual void invalid_values(vector<size_t>& /*index*/, 
                              vector<double>& /*value*/) {
    throw std::runtime_error("valid_values() not implemented");
  }
};

template<class T>
class AgradDistributionTestFixture : public ::testing::Test {
public:
  /*vector<double> first_valid_params() {
    vector<vector<double> > params;
    vector<double> log_prob;
    T().valid_values(params, log_prob); 
    return params[0];
  }
  double e() {
    return 1e-8;
    }*/
};
TYPED_TEST_CASE_P(AgradDistributionTestFixture);


TYPED_TEST_P(AgradDistributionTestFixture, DoesBlah) {
  FAIL();
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
                           DoesBlah);





class AgradCdfTest {
};

template<class T>
class AgradCdfTestFixture : public ::testing::Test {
};

TYPED_TEST_CASE_P(AgradCdfTestFixture);


TYPED_TEST_P(AgradCdfTestFixture, DoesBlah) {
  FAIL();
}

REGISTER_TYPED_TEST_CASE_P(AgradCdfTestFixture,
                           DoesBlah);


#endif
