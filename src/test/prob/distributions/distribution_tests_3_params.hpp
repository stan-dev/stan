#ifndef __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___
#define __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TESTS_3_PARAMS_HPP___

TYPED_TEST_P(DistributionTestFixture, check_valid) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0);

  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    double expected_value = expected_values[n];

    EXPECT_FLOAT_EQ(expected_value,
		    _LOG_PROB_<false>(params[0],
				      params[1],
				      params[2]))
      << "Failed at index: " << n << std::endl;
    
    EXPECT_FLOAT_EQ(0.0,
		    _LOG_PROB_<true>(params[0],
				     params[1],
				     params[2]))
      << "Failed at index: " << n << std::endl;
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
      << "Failed at index: " << n << std::endl;

    double expected_log_prob = 0.0;
    EXPECT_NO_THROW(expected_log_prob = _LOG_PROB_<false>(invalid_params[0], 
							  invalid_params[1],
							  invalid_params[2],
							  errno_policy()))
      << "errno policy. "
      << "Failed at index: " << n << std::endl;
    
    EXPECT_TRUE(std::isnan(expected_log_prob))
      << "errno policy. "
      << "Failed at index: " << n << std::endl;
  }

  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<false>(invalid_params[0], 
				   invalid_params[1],
				   invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i;

    double expected_log_prob = 0.0;
    EXPECT_NO_THROW(expected_log_prob = _LOG_PROB_<false>(invalid_params[0], 
							  invalid_params[1],
							  invalid_params[2],
							  errno_policy()))
      << "errno policy with NaN for parameter: " << i;
    
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
  ASSERT_GT(parameters.size(), 0);
  
  size_t N = parameters[0].size();
  vector<double> param1, param2, param3;
  for (size_t n = 0; n < parameters.size(); n++) {
    param1.push_back(parameters[n][0]);
    param2.push_back(parameters[n][1]);
    param3.push_back(parameters[n][2]);
  }
  EXPECT_FLOAT_EQ(stan::math::sum(expected_values),
		  _LOG_PROB_<false>(param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_vector) {
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  
  double log_prob = 0.0;
  EXPECT_NO_THROW(log_prob = _LOG_PROB_<false>(param1, param2, param3, errno_policy()));
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_matrix) {
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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
  TypeParam t;
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

#endif
