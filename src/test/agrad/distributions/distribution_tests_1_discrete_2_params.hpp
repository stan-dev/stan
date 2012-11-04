#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_2_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_2_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>

// i: int
// I: vector

using stan::agrad::var;

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();
  ASSERT_EQ(3U, parameters.size());

  
  int param0;
  var param1, param2;
  var logprob;
  param0 = int(parameters[0]);
  param1 = parameters[1];
  param2 = parameters[2];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, param2, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_idd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    ASSERT_EQ(3U, params.size());
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]),
					  params[1],
					  params[2]))
      << "Failed with (i,d,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ", " << params[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (i,d,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_idv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]), 
					  params[1],
					  var(params[2])))
      << "Failed with (i,d,v) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ivd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]),
					  var(params[1]),
					  params[2]))
      << "Failed with (i,v,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_ivv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]),
					  var(params[1]),
					  var(params[2])))
      << "Failed with (i,d,v) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ", " << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_idd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]), 
				  invalid_params[1],
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]), 
				  invalid_params[1], 
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_idv) {  
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  invalid_params[1],
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]), 
				  invalid_params[1], 
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ivd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  var(invalid_params[1]),
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  var(invalid_params[1]), 
				  invalid_params[2]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_ivv) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  var(invalid_params[1]),
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  var(invalid_params[1]), 
				  var(invalid_params[2])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << ", " << invalid_params[1] << ", " << invalid_params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_idd) {
  var logprob_true = _LOG_PROB_<true>(int(this->first_valid_params()[0]),
				      this->first_valid_params()[1],
				      this->first_valid_params()[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(int(params[0]), 
					 params[1],
					 params[2]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_idv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(int(params[0]), 
					params[1],
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(int(params[0]), 
				      params[1],
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(int(params[0]),
					   params[1],
					   var(params[2]));
    var logprob2_true = _LOG_PROB_<true>(int(params[0]),
					 params[1],
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ivd) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(int(params[0]),
					var(params[1]),
					params[2]);
  var logprob_true = _LOG_PROB_<true>(int(params[0]), 
				      var(params[1]),
				      params[2]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(int(params[0]), 
					   var(params[1]),
					   params[2]);
    var logprob2_true = _LOG_PROB_<true>(int(params[0]), 
					 var(params[1]),
					 params[2]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_ivv) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(int(params[0]), 
					var(params[1]),
					var(params[2]));
  var logprob_true = _LOG_PROB_<true>(int(params[0]), 
				      var(params[1]),
				      var(params[2]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    params[2] = parameters[n][2];
    var logprob2_false = _LOG_PROB_<false>(int(params[0]), 
					   var(params[1]),
					   var(params[2]));
    var logprob2_true = _LOG_PROB_<true>(int(params[0]), 
					 var(params[1]),
					 var(params[2]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << this->first_valid_params()[1] << "," << this->first_valid_params()[2] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << params[1] << "," << params[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_idd) {
  SUCCEED() << "No op for (i,d,d)" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_idv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g2 = (_LOG_PROB_<false>(int(p[0]), p[1], p[2]+e) - _LOG_PROB_<false>(int(p[0]), p[1], p[2]-e)) / e_times_2;
      
    var p2(p[2]);
    var lp = _LOG_PROB_<true>(int(p[0]), p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g2,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 2" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ivd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(int(p[0]), p[1]+e, p[2]) - _LOG_PROB_<false>(int(p[0]), p[1]-e, p[2])) / e_times_2;
      
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p[2]);
    vector<var> v_params(1);
    v_params[0] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_ivv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(int(p[0]), p[1]+e, p[2]) - _LOG_PROB_<false>(int(p[0]), p[1]-e, p[2])) / e_times_2;
    double diff_g2 = (_LOG_PROB_<false>(int(p[0]), p[1], p[2]+e) - _LOG_PROB_<false>(int(p[0]), p[1], p[2]-e)) / e_times_2;
      
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p2);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_NEAR(diff_g2,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_idd) {
  SUCCEED() << "No op for (i,d,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_idv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p[1], p2);
    var expected_lp = TypeParam().log_prob(int(p[0]), p[1], p2);
    vector<var> v_params(1);
    v_params[0] = p2;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ivd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p[2]);
    var expected_lp = TypeParam().log_prob(int(p[0]), p1, p[2]);
    vector<var> v_params(1);
    v_params[0] = p1;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_ivv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    var p2(p[2]);
    
    var lp = _LOG_PROB_<true>(int(p[0]), p1, p2);
    var expected_lp = TypeParam().log_prob(int(p[0]), p1, p2);
    vector<var> v_params(2);
    v_params[0] = p1;
    v_params[1] = p2;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << int(p[0]) << ", " << p[1] << ", " << p[2] << ")" << std::endl;
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IVV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p1, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    var p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2.push_back(grad[1]);
  }

  vector<int> p0;
  vector<var> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  logprob.grad(p1, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IVD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    var p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<int> p0;
  vector<var> p1;
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IVv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    var p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2 += grad[1];
  }

  vector<int> p0;
  vector<var> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad;
  vector<var> x(p1);
  x.push_back(p2);
  logprob.grad(x, grad);

  vector<double> grad_p1;
  double grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    grad_p1.push_back(grad[n]);
  }
  grad_p2 = grad[parameters.size()];
  
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2)
    << "Gradient failed for parameter 2"; 
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IVd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    var p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<int> p0;
  vector<var> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IDV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    double p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<int> p0;
  vector<double> p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IDv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    double p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  vector<int> p0;
  vector<double> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IvV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;

    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
    expected_grad_p2.push_back(grad[1]);
  }
  
  vector<int> p0;
  var p1 = parameters[0][1];
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad;
  vector<var> x;
  x.push_back(p1);
  x.insert(x.end(), p2.begin(), p2.end());
  logprob.grad(x, grad);

  double grad_p1;
  vector<double> grad_p2;
  grad_p1 = grad[0];
  for (size_t n = 0; n < parameters.size(); n++)
    grad_p2.push_back(grad[n+1]);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1) 
    << "Gradient failed for parameter 1"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IvD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[n][0];
    var p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  vector<int> p0;
  var p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0]) 
    << "Gradient failed for parameter 1"; 
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IdV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    double p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  vector<int> p0;
  double p1 = parameters[0][1];
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iVV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p1, expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    var p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2.push_back(grad[1]);
  }

  int p0 = int(parameters[0][0]);
  vector<var> p1, p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1, grad_p2;
  logprob.grad(p1, grad_p1);
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iVD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    var p1 = parameters[n][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  int p0 = int(parameters[0][0]);
  vector<var> p1;
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iVv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    var p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
    expected_grad_p2 += grad[1];
  }

  int p0 = int(parameters[0][0]);
  vector<var> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad;
  vector<var> x(p1);
  x.push_back(p2);
  logprob.grad(x, grad);

  vector<double> grad_p1;
  double grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    grad_p1.push_back(grad[n]);
  }
  grad_p2 = grad[parameters.size()];
  
  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2)
    << "Gradient failed for parameter 2"; 
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iVd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    var p1 = parameters[n][1];
    double p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  int p0 = int(parameters[0][0]);
  vector<var> p1;
  double p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n]) 
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iDV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    double p1 = parameters[n][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  int p0 = int(parameters[0][0]);
  vector<double> p1;
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iDv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p2 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    double p1 = parameters[n][1];
    var p2 = parameters[0][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2 += grad[0];
  }

  int p0 = int(parameters[0][0]);
  vector<double> p1;
  var p2 = parameters[0][2];
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  vector<var> x;
  x.push_back(p2);
  logprob.grad(x, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p2, grad_p2[0])
    << "Gradient failed for parameter 2"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_ivV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[0][0];
    var p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;

    x.push_back(p1);
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
    expected_grad_p2.push_back(grad[1]);
  }
  
  int p0 = int(parameters[0][0]);
  var p1 = parameters[0][1];
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad;
  vector<var> x;
  x.push_back(p1);
  x.insert(x.end(), p2.begin(), p2.end());
  logprob.grad(x, grad);

  double grad_p1;
  vector<double> grad_p2;
  grad_p1 = grad[0];
  for (size_t n = 0; n < parameters.size(); n++)
    grad_p2.push_back(grad[n+1]);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1) 
    << "Gradient failed for parameter 1"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_ivD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[0][0];
    var p1 = parameters[0][1];
    double p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  int p0 = int(parameters[0][0]);
  var p1 = parameters[0][1];
  vector<double> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0]) 
    << "Gradient failed for parameter 1"; 
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_idV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[0][0]);
    double p1 = parameters[0][1];
    var p2 = parameters[n][2];
    var logprob = _LOG_PROB_<true>(p0, p1, p2);
    vector<var> x;
    x.push_back(p2);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p2.push_back(grad[0]);
  }

  int p0 = int(parameters[0][0]);
  double p1 = parameters[0][1];
  vector<var> p2;
  for (size_t n = 0; n < parameters.size(); n++) {
    p2.push_back(parameters[n][2]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1, p2);
  vector<double> grad_p2;
  logprob.grad(p2, grad_p2);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p2[n], grad_p2[n])
      << "Index " << n << ": gradient failed for parameter 2"; 
  }
}






// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_idd,
			   check_valid_idv,
			   check_valid_ivd,
			   check_valid_ivv,
			   check_invalid_idd,
			   check_invalid_idv,
			   check_invalid_ivd,
			   check_invalid_ivv,
			   logprob_propto_idd,
			   logprob_propto_idv,
			   logprob_propto_ivd,
			   logprob_propto_ivv,
			   gradient_finite_diff_idd,
			   gradient_finite_diff_idv,
			   gradient_finite_diff_ivd,
			   gradient_finite_diff_ivv,
			   gradient_function_idd,
			   gradient_function_idv,
			   gradient_function_ivd,
			   gradient_function_ivv,
			   vectorized_IVV,
			   vectorized_IVD,
			   vectorized_IVv,
			   vectorized_IVd,
			   vectorized_IDV,
			   vectorized_IDv,
			   vectorized_IvV,
			   vectorized_IvD,
			   vectorized_IdV,
			   vectorized_iVV,
			   vectorized_iVD,
			   vectorized_iVv,
			   vectorized_iVd,
			   vectorized_iDV,
			   vectorized_iDv,
			   vectorized_ivV,
			   vectorized_ivD,
			   vectorized_idV);
#endif
