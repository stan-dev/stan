#ifndef __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TEST_FIXTURE_HPP___
#define __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TEST_FIXTURE_HPP___

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

//template <typename... Args>
//double log_prob(Args&&... args);

/*template <typename T1, typename T2, class Policy>
double log_prob(const T1&, 
		const T2&,
		Policy&);*/

// 3 arguments
template <typename T1, typename T2, typename T3, class Policy>
double log_prob(const T1&,
		const T2&,
		const T3&,
		Policy&);
  
/*// 4 arguments
template <class Policy>
double log_prob(const vector<double>&,
		const vector<double>&,
		const vector<double>&,
		const vector<double>&,
		Policy&);*/

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

  /*
  // 2 arguments
  template <typename T1, typename T2, class Policy>
  virtual double log_prob(const T1&, 
			  const T2&,
			  Policy&) {
    return 0;
  }
  
  // 3 arguments
  template <typename T1, typename T2, typename T3, class Policy>
  virtual double log_prob(const T1&,
			  const T2&,
			  const T3&,
			  Policy&) {
    return 0;
  }
  
  // 4 arguments
  template <class Policy>
  virtual double log_prob(const vector<double>&,
			  const vector<double>&,
			  const vector<double>&,
			  const vector<double>&,
			  Policy&) {
    return 0;
    }
  */
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
};

TYPED_TEST_CASE_P(DistributionTestFixture);

TYPED_TEST_P(DistributionTestFixture, valid_default_policy) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0);

  size_t N = parameters[0].size();
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    double expected_value = expected_values[n];
    /*if (N == 2)
      EXPECT_FLOAT_EQ(expected_value,
		      //t.log_prob(params[0], 
		      log_prob(params[0], 
			       params[1],
			       default_policy()))
			       << "Failed at index: " << n << std::endl;*/
    if (N == 3)
      EXPECT_FLOAT_EQ(expected_value, log_prob(0.0, 0.0, 0.0, default_policy()));
		      //log_prob(params[0], 
		      //params[1], 
		      //params[2],
		      //default_policy()))
	<< "Failed at index: " << n << std::endl;
    /*if (N == 4)
      EXPECT_FLOAT_EQ(expected_value,
		      log_prob(params[0], 
				 params[1], 
				 params[2], 
				 params[3],
				 default_policy()))
	<< "Failed at index: " << n << std::endl;
    */
  }
}

TYPED_TEST_P(DistributionTestFixture, invalid_default_policy) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;
    
  const vector<double> valid_params = this->first_valid_params();
  const size_t N = valid_params.size();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    /*if (N == 2)
      EXPECT_THROW(log_prob(invalid_params[0], 
			      invalid_params[1],
			      default_policy()),
		   std::domain_error)
		   << "Failed at index: " << n << std::endl;*/
    if (N == 3)
      EXPECT_THROW(log_prob(invalid_params[0], 
			      invalid_params[1],
			      invalid_params[2],
			      default_policy()),
		   std::domain_error)
	<< "Failed at index: " << n << std::endl;
    /*if (N == 4)
      EXPECT_THROW(log_prob(invalid_params[0], 
			      invalid_params[1],
			      invalid_params[2],
			      invalid_params[3],
			      default_policy()),
		   std::domain_error)
		   << "Failed at index: " << n << std::endl;*/
  }
}

TYPED_TEST_P(DistributionTestFixture, invalid_errno_policy) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;
    
  const vector<double> valid_params = this->first_valid_params();
  const size_t N = valid_params.size();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    double expected_log_prob;    
    /*if (N == 2)
      EXPECT_NO_THROW(log_prob = log_prob(invalid_params[0], 
					    invalid_params[1],
					    errno_policy()))
					    << "Failed at index: " << n << std::endl;*/
    if (N == 3)
      EXPECT_NO_THROW(expected_log_prob = log_prob(invalid_params[0], 
					    invalid_params[1],
					    invalid_params[2],
					    errno_policy()))
	<< "Failed at index: " << n << std::endl;
    /*if (N == 4)
      EXPECT_NO_THROW(log_prob = log_prob(invalid_params[0], 
					    invalid_params[1],
					    invalid_params[2],
					    invalid_params[3],
					    errno_policy()))
					    << "Failed at index: " << n << std::endl;*/
    
    EXPECT_TRUE(std::isnan(expected_log_prob))
      << "Failed at index: " << n << std::endl;
  }
}

TYPED_TEST_P(DistributionTestFixture, valid_vector) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;
  
  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0);
  
  size_t N = parameters[0].size();
  vector<double> param1, param2, param3, param4;
  for (size_t n = 0; n < parameters.size(); n++) {
    if (N > 0)
      param1.push_back(parameters[n][0]);
    if (N > 1)
      param2.push_back(parameters[n][1]);
    if (N > 2)
      param3.push_back(parameters[n][2]);
    if (N > 3)
      param4.push_back(parameters[n][3]);
  }
  /*if (N == 2)
    EXPECT_FLOAT_EQ(stan::math::sum(expected_values),
    log_prob(param1, param2, default_policy()));*/
  if (N == 3)
    EXPECT_FLOAT_EQ(stan::math::sum(expected_values),
		    log_prob(param1, param2, param3, default_policy()));
  /*if (N == 4)
    EXPECT_FLOAT_EQ(stan::math::sum(expected_values),
    log_prob(param1, param2, param3, param4, default_policy()));*/
}

TYPED_TEST_P(DistributionTestFixture, valid_vector_different_sizes) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;
  size_t N_repeat = 10;
  
  t.valid_values(parameters, expected_values);
  
  size_t N = parameters[0].size();
  vector<double> params = parameters[0];
  vector<double> param1, param2, param3, param4;

  if (N > 0)
    param1.assign(N_repeat, params[2]);
  if (N > 1) 
    param2.push_back(params[1]);
  if (N > 2) 
    param3.assign(N_repeat, params[2]);
  if (N > 3) 
    param4.push_back(params[3]);
  
  /*if (N == 2)
    EXPECT_FLOAT_EQ(expected_values[0] * N_repeat,
    log_prob(param1, param2, default_policy()));*/
  if (N == 3)
    EXPECT_FLOAT_EQ(expected_values[0] * N_repeat,
		    log_prob(param1, param2, param3, default_policy()));
  /*if (N == 4)
    EXPECT_FLOAT_EQ(expected_values[0] * N_repeat,
    log_prob(param1, param2, param3, param4, default_policy()));*/
  
  std::cout << "N: " << N << std::endl;
  
  std::cout << "params " << params[0];
  for (size_t n = 1; n < params.size(); n++) {
    std::cout << ", " << params[n];
  }
  std::cout  << std::endl;
  
  std::cout << "param1 " << param1[0];
  for (size_t n = 1; n < param1.size(); n++) {
    std::cout << ", " << param1[n];
  }
  std::cout  << std::endl;

  std::cout << "param2 " << param2[0];
  for (size_t n = 1; n < param2.size(); n++) {
    std::cout << ", " << param2[n];
  }
  std::cout  << std::endl;

  std::cout << "param3 " << param3[0];
  for (size_t n = 1; n < param3.size(); n++) {
    std::cout << ", " << param3[n];
  }
  std::cout  << std::endl;
}


REGISTER_TYPED_TEST_CASE_P(DistributionTestFixture,
			   valid_default_policy,
			   invalid_default_policy,
			   invalid_errno_policy,
			   valid_vector,
			   valid_vector_different_sizes);

#endif
