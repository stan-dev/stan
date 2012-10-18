#ifndef __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___
#define __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___

TYPED_TEST_P(DistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();
  ASSERT_EQ(3U, parameters.size());
  double param1, param2, param3;
  double logprob;
  param1 = parameters[0];
  param2 = parameters[1];
  param3 = parameters[2];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, param3, errno_policy()));
}

TYPED_TEST_P(DistributionTestFixture, check_valid) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0U);

  for (size_t n = 0; n < parameters.size(); n++) {
    ASSERT_EQ(3U, parameters[n].size());
    vector<double> params = parameters[n];
    double expected_value = expected_values[n];

    EXPECT_FLOAT_EQ(expected_value,
		    _LOG_PROB_<false>(params[0],
				      params[1],
				      params[2]))
      << "Failed at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
    
    EXPECT_FLOAT_EQ(0.0,
		    _LOG_PROB_<true>(params[0],
				     params[1],
				     params[2]))
      << "Failed at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}

TYPED_TEST_P(DistributionTestFixture, check_invalid) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;
    
  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<false>(invalid_params[0], 
				   invalid_params[1],
				   invalid_params[2]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;


    double expected_log_prob = 0.0;
    EXPECT_NO_THROW(expected_log_prob = _LOG_PROB_<false>(invalid_params[0], 
							  invalid_params[1],
							  invalid_params[2],
							  errno_policy()))
      << "errno policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
    
    EXPECT_TRUE(std::isnan(expected_log_prob))
      << "errno policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }

  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<false>(invalid_params[0], 
				   invalid_params[1],
				   invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;

    double expected_log_prob = 0.0;
    EXPECT_NO_THROW(expected_log_prob = _LOG_PROB_<false>(invalid_params[0], 
							  invalid_params[1],
							  invalid_params[2],
							  errno_policy()))
      << "errno policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
    
    EXPECT_TRUE(std::isnan(expected_log_prob))
      << "errno policy with NaN for parameter: " << i;
  }
}

TYPED_TEST_P(DistributionTestFixture, valid_vector) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;
  
  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0U);
  
  vector<double> param1, param2, param3;
  for (size_t n = 0; n < parameters.size(); n++) {
    ASSERT_EQ(3U, parameters[n].size());
    param1.push_back(parameters[n][0]);
    param2.push_back(parameters[n][1]);
    param3.push_back(parameters[n][2]);
  }
  EXPECT_FLOAT_EQ(stan::math::sum(expected_values),
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1, param2, param3;
  param1.assign(N_repeat, parameters[0]);
  param2.assign(N_repeat, parameters[1]);
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1, param2;
  double param3;
  param1.assign(N_repeat, parameters[0]);
  param2.assign(N_repeat, parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_double_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1, param3;
  double param2;
  param1.assign(N_repeat, parameters[0]);
  param2 = parameters[1];
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1;
  double param2, param3;
  param1.assign(N_repeat, parameters[0]);
  param2 = parameters[1];
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, double_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param2, param3;
  double param1;
  param1 = parameters[0];
  param2.assign(N_repeat, parameters[1]);
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, double_vector_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param2;
  double param1, param3;
  param1 = parameters[0];
  param2.assign(N_repeat, parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, double_double_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param3;
  double param1, param2;
  param1 = parameters[0];
  param2 = parameters[1];
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}


TYPED_TEST_P(DistributionTestFixture, invalid_different_vector_sizes) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  
  vector<double> params = parameters[0];
  vector<double> param1, param2, param3;

  param1.assign(10U, params[0]);
  param2.assign(1U,  params[1]);
  param3.assign(10U, params[2]);
  
  EXPECT_THROW(_LOG_PROB_<false>(param1, param2, param3), std::domain_error)
    << "Should throw error with sizes (" 
    << param1.size() << ", " << param2.size() << ", " << param3.size() << ")";
  
  double log_prob;
  (void)log_prob;
  EXPECT_NO_THROW(log_prob = _LOG_PROB_<false>(param1, param2, param3, errno_policy()));

  EXPECT_TRUE(std::isnan(log_prob))
    << "errno policy. ";
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat), param2(N_repeat), param3(N_repeat);
  param1.setConstant(parameters[0]);
  param2.setConstant(parameters[1]);
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat), param2(N_repeat);
  double param3;
  param1.setConstant(parameters[0]);
  param2.setConstant(parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_double_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat), param3(N_repeat);
  double param2(N_repeat);
  param1.setConstant(parameters[0]);
  param2 = parameters[1];
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat);
  double param2, param3;
  param1.setConstant(parameters[0]);
  param2 = parameters[1];
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, double_matrix_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  double param1;
  Matrix<double,Dynamic,1> param2(N_repeat), param3(N_repeat);
  param1 = parameters[0];
  param2.setConstant(parameters[1]);
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, double_matrix_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  double param1, param3;
  Matrix<double,Dynamic,1> param2(N_repeat);
  param1 = parameters[0];
  param2.setConstant(parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, double_double_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  double param1, param2;
  Matrix<double,Dynamic,1> param3(N_repeat);
  param1 = parameters[0];
  param2 = parameters[1];
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
		  _LOG_PROB_<false>(param1, param2, param3));
}


REGISTER_TYPED_TEST_CASE_P(DistributionTestFixture,
			   call_all_versions,
			   check_valid,
			   check_invalid,
			   valid_vector,
			   vector_vector_vector,
			   vector_vector_double,
			   vector_double_vector,
			   vector_double_double,
			   double_vector_vector,
			   double_vector_double,
			   double_double_vector,
			   invalid_different_vector_sizes,
			   matrix_matrix_matrix,
			   matrix_matrix_double,
			   matrix_double_matrix,
			   matrix_double_double,
			   double_matrix_matrix,
			   double_matrix_double,
			   double_double_matrix);


#ifdef _CDF_

TYPED_TEST_P(CumulativeTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();
  ASSERT_EQ(3U, parameters.size());
  double param1, param2, param3;
  double cdf;
  param1 = parameters[0];
  param2 = parameters[1];
  param3 = parameters[2];
  
  EXPECT_NO_THROW(cdf = _CDF_(param1, param2, param3));
  EXPECT_NO_THROW(cdf = _CDF_(param1, param2, param3, errno_policy()));
}

TYPED_TEST_P(CumulativeTestFixture, check_valid) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0U);

  for (size_t n = 0; n < parameters.size(); n++) {
    ASSERT_EQ(3U, parameters[n].size());
    vector<double> params = parameters[n];
    double expected_cdf = expected_values[n];

    ASSERT_TRUE(expected_cdf > 0) 
      << "expected cdf value for index " << n << " should be greater than 0";
    ASSERT_TRUE(expected_cdf < 1) 
      << "expected cdf value for index " << n << " should be less than 1";
    
    EXPECT_FLOAT_EQ(expected_cdf,
		    _CDF_(params[0],
			  params[1],
			  params[2]))
      << "Failed at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}

TYPED_TEST_P(CumulativeTestFixture, check_zero) {
  TypeParam t;
  vector<vector<double> > parameters;

  t.zero_values(parameters);
  
  if (parameters.size() == 0)
    return;
  
  ASSERT_GT(parameters.size(), 0U);

  for (size_t n = 0; n < parameters.size(); n++) {
    ASSERT_EQ(3U, parameters[n].size());
    vector<double> params = parameters[n];

    EXPECT_FLOAT_EQ(0.0,
		    _CDF_(params[0],
			  params[1],
			  params[2]))
      << "Failed at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}

TYPED_TEST_P(CumulativeTestFixture, check_one) {
  TypeParam t;
  vector<vector<double> > parameters;

  t.one_values(parameters);

  if (parameters.size() == 0)
    return;
  
  ASSERT_GT(parameters.size(), 0U);

  for (size_t n = 0; n < parameters.size(); n++) {
    ASSERT_EQ(3U, parameters[n].size());
    vector<double> params = parameters[n];
    
    EXPECT_FLOAT_EQ(1.0,
		    _CDF_(params[0],
			  params[1],
			  params[2]))
      << "Failed at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}


TYPED_TEST_P(CumulativeTestFixture, check_invalid) {
  TypeParam t;
  vector<size_t> index;
  vector<double> invalid_values;
    
  const vector<double> valid_params = this->first_valid_params();
  t.invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_CDF_(invalid_params[0], 
		       invalid_params[1],
		       invalid_params[2]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;


    double expected_cdf = 0.0;
    EXPECT_NO_THROW(expected_cdf = _CDF_(invalid_params[0], 
					 invalid_params[1],
					 invalid_params[2],
					 errno_policy()))
      << "errno policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
    
    EXPECT_TRUE(std::isnan(expected_cdf))
      << "errno policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }

  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_CDF_(invalid_params[0], 
		       invalid_params[1],
		       invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;

    double expected_cdf = 0.0;
    EXPECT_NO_THROW(expected_cdf = _CDF_(invalid_params[0], 
					 invalid_params[1],
					 invalid_params[2],
					 errno_policy()))
      << "errno policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
    
    EXPECT_TRUE(std::isnan(expected_cdf))
      << "errno policy with NaN for parameter: " << i;
  }
}

TYPED_TEST_P(CumulativeTestFixture, vector_vector_vector) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1, param2, param3;
  param1.assign(N_repeat, parameters[0]);
  param2.assign(N_repeat, parameters[1]);
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, vector_vector_double) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1, param2;
  double param3;
  param1.assign(N_repeat, parameters[0]);
  param2.assign(N_repeat, parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, vector_double_vector) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1, param3;
  double param2;
  param1.assign(N_repeat, parameters[0]);
  param2 = parameters[1];
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, vector_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<double> param1;
  double param2, param3;
  param1.assign(N_repeat, parameters[0]);
  param2 = parameters[1];
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, double_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  double param1;
  vector<double> param2, param3;
  param1 = parameters[0];
  param2.assign(N_repeat, parameters[1]);
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, double_vector_double) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  double param1, param3;
  vector<double> param2;
  param1 = parameters[0];
  param2.assign(N_repeat, parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, double_double_vector) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  double param1, param2;
  vector<double> param3;
  param1 = parameters[0];
  param2 = parameters[1];
  param3.assign(N_repeat, parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, invalid_different_vector_sizes) { 
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  
  vector<double> params = parameters[0];
  vector<double> param1, param2, param3;

  param1.assign(10U, params[0]);
  param2.assign(1U,  params[1]);
  param3.assign(10U, params[2]);
  
  EXPECT_THROW(_CDF_(param1, param2, param3), std::domain_error)
    << "Should throw error with sizes (" 
    << param1.size() << ", " << param2.size() << ", " << param3.size() << ")";
}
TYPED_TEST_P(CumulativeTestFixture, matrix_matrix_matrix) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat), param2(N_repeat), param3(N_repeat);
  param1.setConstant(parameters[0]);
  param2.setConstant(parameters[1]);
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, matrix_matrix_double) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat), param2(N_repeat);
  double param3;
  param1.setConstant(parameters[0]);
  param2.setConstant(parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, matrix_double_matrix) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat), param3(N_repeat);
  double param2;
  param1.setConstant(parameters[0]);
  param2 = parameters[1];
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, matrix_double_double) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<double,Dynamic,1> param1(N_repeat);
  double param2, param3;
  param1.setConstant(parameters[0]);
  param2 = parameters[1];
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, double_matrix_matrix) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  double param1;
  Matrix<double,Dynamic,1> param2(N_repeat), param3(N_repeat);
  param1 = parameters[0];
  param2.setConstant(parameters[1]);
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, double_matrix_double) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  double param1, param3;
  Matrix<double,Dynamic,1> param2(N_repeat);
  param1 = parameters[0];
  param2.setConstant(parameters[1]);
  param3 = parameters[2];
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}
TYPED_TEST_P(CumulativeTestFixture, double_double_matrix) { 
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  double param1, param2;
  Matrix<double,Dynamic,1> param3(N_repeat);
  param1 = parameters[0];
  param2 = parameters[1];
  param3.setConstant(parameters[2]);
  EXPECT_FLOAT_EQ(pow(expected_value, N_repeat),
		  _CDF_(param1, param2, param3));
}

REGISTER_TYPED_TEST_CASE_P(CumulativeTestFixture,
			   call_all_versions,
			   check_valid,
			   check_zero,
			   check_one,
			   check_invalid,
			   vector_vector_vector,
			   vector_vector_double,
			   vector_double_vector,
			   vector_double_double,
			   double_vector_vector,
			   double_vector_double,
			   double_double_vector,
			   invalid_different_vector_sizes,
			   matrix_matrix_matrix,
			   matrix_matrix_double,
			   matrix_double_matrix,
			   matrix_double_double,
			   double_matrix_matrix,
			   double_matrix_double,
			   double_double_matrix);
#endif

#endif
