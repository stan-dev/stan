#ifndef __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_2_PARAMS_HPP___
#define __TEST__PROB__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_DISCRETE_2_PARAMS_HPP___

TYPED_TEST_P(DistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();
  ASSERT_EQ(4U, parameters.size());
  int param0, param1;
  double param2, param3;
  double logprob = 0.0;
  (void) logprob; // supress unused warning
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3 = parameters[3];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, param3, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, param3));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, param3, errno_policy()));
}

TYPED_TEST_P(DistributionTestFixture, check_valid) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  ASSERT_EQ(parameters.size(), expected_values.size());
  ASSERT_GT(parameters.size(), 0U);

  for (size_t n = 0; n < parameters.size(); n++) {
    ASSERT_EQ(4U, parameters[n].size());
    vector<double> params = parameters[n];
    double expected_value = expected_values[n];

    EXPECT_FLOAT_EQ(expected_value,
                    _LOG_PROB_<false>(int(params[0]),
                                      int(params[1]),
                                      params[2],
                                      params[3]))
      << "Failed at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << int(params[1]) << ", " << params[2] << ", " << params[3] << ")" << std::endl;
    
    EXPECT_FLOAT_EQ(0.0,
                    _LOG_PROB_<true>(int(params[0]),
                                     int(params[1]),
                                     params[2],
                                     params[3]))
      << "Failed at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << int(params[1]) << ", " << params[2] << ", " << params[3] << ")" << std::endl;
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
    ASSERT_TRUE(index[n] < 4);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<false>(int(invalid_params[0]), 
                                   int(invalid_params[1]),
                                   invalid_params[2],
                                   invalid_params[3]),
                 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << int(invalid_params[1]) << ", " << invalid_params[2] << ", " << invalid_params[3] << ")" << std::endl;

    double expected_log_prob = 0.0;
    EXPECT_NO_THROW(expected_log_prob = _LOG_PROB_<false>(int(invalid_params[0]), 
                                                          int(invalid_params[1]),
                                                          invalid_params[2],
                                                          invalid_params[3],
                                                          errno_policy()))
      << "errno policy. "
      << "Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << int(invalid_params[1]) << ", " << invalid_params[2] << ", " << invalid_params[3] << ")" << std::endl;
    
    EXPECT_TRUE(std::isnan(expected_log_prob))
      << "errno policy. "
      << "Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << int(invalid_params[1]) << ", " << invalid_params[2] << ", " << invalid_params[3] << ")" << std::endl;
  }

  for (size_t i = 2; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<false>(int(invalid_params[0]), 
                                   int(invalid_params[1]),
                                   invalid_params[2],
                                   invalid_params[3]),
                 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << ", " << int(invalid_params[1]) << ", " << invalid_params[2] << ", " << invalid_params[3] << ")" << std::endl;

    double expected_log_prob = 0.0;
    EXPECT_NO_THROW(expected_log_prob = _LOG_PROB_<false>(int(invalid_params[0]), 
                                                          int(invalid_params[1]),
                                                          invalid_params[2],
                                                          invalid_params[3],
                                                          errno_policy()))
      << "errno policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << ", " << int(invalid_params[1]) << ", " << invalid_params[2] << ", " << invalid_params[3] << ")" << std::endl;
    
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

  vector<int> param0, param1;
  vector<double> param2, param3;
  for (size_t n = 0; n < parameters.size(); n++) {
    param0.push_back(int(parameters[n][0]));
    param1.push_back(int(parameters[n][1]));
    param2.push_back(parameters[n][2]);
    param3.push_back(parameters[n][3]);
  }
  EXPECT_FLOAT_EQ(stan::math::sum(expected_values),
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<int> param0, param1;
  vector<double> param2, param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1.assign(N_repeat, int(parameters[1]));
  param2.assign(N_repeat, parameters[2]);
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_vector_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<int> param0, param1;
  vector<double> param2;
  double param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1.assign(N_repeat, int(parameters[1]));
  param2.assign(N_repeat, parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_double_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;

  vector<int> param0, param1;
  double param2;
  vector<double> param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1.assign(N_repeat, int(parameters[1]));
  param2 = parameters[2];
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_vector_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;

  vector<int> param0, param1;
  double param2, param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1.assign(N_repeat, int(parameters[1]));
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_int_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<int> param0;
  int param1;
  vector<double> param2, param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1 = int(parameters[1]);
  param2.assign(N_repeat, parameters[2]);
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_int_vector_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<int> param0;
  int param1;
  vector<double> param2;
  double param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1 = int(parameters[1]);
  param2.assign(N_repeat, parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_int_double_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<int> param0;
  int param1;
  double param2;
  vector<double> param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, vector_int_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  vector<int> param0;
  int param1;
  double param2, param3;
  param0.assign(N_repeat, int(parameters[0]));
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_vector_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  int param0;
  vector<int> param1;
  vector<double> param2, param3;
  param0 = int(parameters[0]);
  param1.assign(N_repeat, int(parameters[1]));
  param2.assign(N_repeat, parameters[2]);
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_vector_vector_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  int param0;
  vector<int> param1;
  vector<double> param2;
  double param3;
  param0 = int(parameters[0]);
  param1.assign(N_repeat, int(parameters[1]));
  param2.assign(N_repeat, parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_vector_double_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;

  int param0;
  vector<int> param1;
  double param2;
  vector<double> param3;
  param0 = int(parameters[0]);
  param1.assign(N_repeat, int(parameters[1]));
  param2 = parameters[2];
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_vector_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;

  int param0;
  vector<int> param1;
  double param2, param3;
  param0 = int(parameters[0]);
  param1.assign(N_repeat, int(parameters[1]));
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_vector_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  int param0, param1;
  vector<double> param2, param3;
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2.assign(N_repeat, parameters[2]);
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_vector_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  int param0, param1;
  vector<double> param2;
  double param3;
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2.assign(N_repeat, parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_double_vector) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 10;
  int param0, param1;
  double param2;
  vector<double> param3;
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3.assign(N_repeat, parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  int param0, param1;
  double param2, param3;
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}
TYPED_TEST_P(DistributionTestFixture, invalid_different_vector_sizes) {
  TypeParam t;
  vector<vector<double> > parameters;
  vector<double> expected_values;

  t.valid_values(parameters, expected_values);
  
  vector<double> params = parameters[0];
  vector<int> param0, param1;
  vector<double> param2, param3;

  param0.assign(10U, int(params[0]));
  param1.assign(10U, int(params[1]));
  param2.assign(10U, params[2]);
  param3.assign(1U,  params[3]);
  
  EXPECT_THROW(_LOG_PROB_<false>(param0, param1, param2, param3), std::domain_error)
    << "Should throw error with sizes (" 
    << param0.size() << ", " << param1.size() << ", " << param2.size() << ", " << param3.size() << ")";
  
  double log_prob = 0.0;
  EXPECT_NO_THROW(log_prob = _LOG_PROB_<false>(param0, param1, param2, param3, errno_policy()));
  
  EXPECT_TRUE(std::isnan(log_prob))
    << "errno policy. ";
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_matrix_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat), param1(N_repeat);
  Matrix<double,Dynamic,1> param2(N_repeat), param3(N_repeat);
  param0.setConstant(int(parameters[0]));
  param1.setConstant(int(parameters[1]));
  param2.setConstant(parameters[2]);
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_matrix_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat), param1(N_repeat);
  Matrix<double,Dynamic,1> param2(N_repeat);
  double param3;
  param0.setConstant(int(parameters[0]));
  param1.setConstant(int(parameters[1]));
  param2.setConstant(parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_double_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat), param1(N_repeat);
  double param2;
  Matrix<double,Dynamic,1> param3(N_repeat);
  param0.setConstant(int(parameters[0]));
  param1.setConstant(int(parameters[1]));
  param2 = parameters[2];
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_matrix_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat), param1(N_repeat);
  double param2, param3;
  param0.setConstant(int(parameters[0]));
  param1.setConstant(int(parameters[1]));
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_int_matrix_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat);
  int param1;
  Matrix<double,Dynamic,1> param2(N_repeat), param3(N_repeat);
  param0.setConstant(int(parameters[0]));
  param1 = int(parameters[1]);
  param2.setConstant(parameters[2]);
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_int_matrix_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat);
  int param1;
  Matrix<double,Dynamic,1> param2(N_repeat);
  double param3(N_repeat);
  param0.setConstant(int(parameters[0]));
  param1 = int(parameters[1]);
  param2.setConstant(parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_int_double_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat);
  int param1;
  double param2;
  Matrix<double,Dynamic,1> param3(N_repeat);
  param0.setConstant(int(parameters[0]));
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, matrix_int_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  Matrix<int,Dynamic,1> param0(N_repeat);
  int param1;
  double param2, param3;
  param0.setConstant(int(parameters[0]));
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_matrix_matrix_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0;
  Matrix<int,Dynamic,1> param1(N_repeat);
  Matrix<double,Dynamic,1> param2(N_repeat), param3(N_repeat);
  param0 = int(parameters[0]);
  param1.setConstant(int(parameters[1]));
  param2.setConstant(parameters[2]);
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_matrix_matrix_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0;
  Matrix<int,Dynamic,1> param1(N_repeat);
  Matrix<double,Dynamic,1> param2(N_repeat);
  double param3;
  param0 = int(parameters[0]);
  param1.setConstant(int(parameters[1]));
  param2.setConstant(parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_matrix_double_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0;
  Matrix<int,Dynamic,1> param1(N_repeat);
  double param2;
  Matrix<double,Dynamic,1> param3(N_repeat);
  param0 = int(parameters[0]);
  param1.setConstant(int(parameters[1]));
  param2 = parameters[2];
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_matrix_double_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0;
  Matrix<int,Dynamic,1> param1(N_repeat);
  double param2, param3;
  param0 = int(parameters[0]);
  param1.setConstant(int(parameters[1]));
  param2 = parameters[2];
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_matrix_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0, param1;
  Matrix<double,Dynamic,1> param2(N_repeat), param3(N_repeat);
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2.setConstant(parameters[2]);
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_matrix_double) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0, param1;
  Matrix<double,Dynamic,1> param2(N_repeat);
  double param3(N_repeat);
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2.setConstant(parameters[2]);
  param3 = parameters[3];
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}

TYPED_TEST_P(DistributionTestFixture, int_int_double_matrix) {
  vector<double> parameters = this->first_valid_params();
  double expected_value = this->first_valid_value();
  
  size_t N_repeat = 25;
  int param0, param1;
  double param2;
  Matrix<double,Dynamic,1> param3(N_repeat);
  param0 = int(parameters[0]);
  param1 = int(parameters[1]);
  param2 = parameters[2];
  param3.setConstant(parameters[3]);
  EXPECT_FLOAT_EQ(N_repeat * expected_value,
                  _LOG_PROB_<false>(param0, param1, param2, param3));
}


REGISTER_TYPED_TEST_CASE_P(DistributionTestFixture,
                           call_all_versions,
                           check_valid,
                           check_invalid,
                           valid_vector,
                           vector_vector_vector_vector,
                           vector_vector_vector_double,
                           vector_vector_double_vector,
                           vector_vector_double_double,
                           vector_int_vector_vector,
                           vector_int_vector_double,
                           vector_int_double_vector,
                           vector_int_double_double,
                           int_vector_vector_vector,
                           int_vector_vector_double,
                           int_vector_double_vector,
                           int_vector_double_double,
                           int_int_vector_vector,
                           int_int_vector_double,
                           int_int_double_vector,
                           int_int_double_double,
                           invalid_different_vector_sizes,
                           matrix_matrix_matrix_matrix,
                           matrix_matrix_matrix_double,
                           matrix_matrix_double_matrix,
                           matrix_matrix_double_double,
                           matrix_int_matrix_matrix,
                           matrix_int_matrix_double,
                           matrix_int_double_matrix,
                           matrix_int_double_double,
                           int_matrix_matrix_matrix,
                           int_matrix_matrix_double,
                           int_matrix_double_matrix,
                           int_matrix_double_double,
                           int_int_matrix_matrix,
                           int_int_matrix_double,
                           int_int_double_matrix);
#endif
