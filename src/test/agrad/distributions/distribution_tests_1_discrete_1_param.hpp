#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_1_PARAM_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_1_DISCRETE_1_PARAM_HPP___

// i: int
// v: var
// d: double
// I: vector<int>
// V: vector<var>
// D: vector<double>

using stan::agrad::var;

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

  int param0;
  var param1;
  var logprob;
  param0 = parameters[0];
  param1 = parameters[1];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param0, param1, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param0, param1, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param0, param1, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_id) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(int(params[0]),
					  params[1]))
      << "Failed with (i,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (i,d) at index: " << n << std::endl
      << "(" << int(params[0]) << ", " << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_iv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1])))
      << "Failed with (i,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << var(params[1]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_id) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  invalid_params[1]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << "," << invalid_params[1] << ")" << std::endl;
  }
  for (size_t i = 1; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]), 
				  invalid_params[1]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << int(invalid_params[0]) << "," << invalid_params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_iv) {
  vector<size_t> index;
  vector<double> invalid_values;
  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]),
				  var(invalid_params[1])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << int(invalid_params[0]) << "," << var(invalid_params[1]) << ")" << std::endl;
  }
  for (size_t i = 1; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(_LOG_PROB_<true>(int(invalid_params[0]), 
				  var(invalid_params[1])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i << std::endl
      << "(" << int(invalid_params[0]) << "," << var(invalid_params[1]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_id) {
  var logprob_true = _LOG_PROB_<true>(int(this->first_valid_params()[0]),
				      this->first_valid_params()[1]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(int(params[0]),
					 params[1]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_iv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(int(params[0]),
					var(params[1]));
  var logprob_true = _LOG_PROB_<true>(int(params[0]),
				      var(params[1]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    var logprob2_false = _LOG_PROB_<false>(int(params[0]),
					   var(params[1]));
    
    var logprob2_true = _LOG_PROB_<true>(int(params[0]),
					 var(params[1]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << int(this->first_valid_params()[0]) << "," << var(this->first_valid_params()[1]) << ") - " 
      << "_LOG_PROB_(" << int(params[0]) << "," << var(params[1]) << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_id) {
  SUCCEED() << "No op for all constants" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_iv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(int(p[0]), p[1]+e) - _LOG_PROB_<false>(int(p[0]), p[1]-e)) / e_times_2;
      
    var p1(p[1]);
    var lp = _LOG_PROB_<true>(p[0], p1);
    vector<var> v_params(1);
    v_params[0] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g1,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_id) {
  SUCCEED() << "No op for (i,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_iv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p[0], p1);
    var expected_lp = TypeParam().log_prob(p[0], p1);
    vector<var> v_params(1);
    v_params[0] = p1;
    
    vector<double> gradients;
    lp.grad(v_params, gradients);
    vector<double> expected_gradients;
    expected_lp.grad(v_params, expected_gradients);
    
    EXPECT_FLOAT_EQ(expected_lp.val(),
		    lp.val())
      << "Index: " << n << " - function value test failed" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[0],
		    gradients[0])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 2" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_IV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<int> p0;
  vector<var> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_ID) {
  SUCCEED() << "No op for (I,D) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Iv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    int p0 = int(parameters[n][0]);
    var p1 = parameters[0][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  vector<int> p0;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(int(parameters[n][0]));
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Id) {
  SUCCEED() << "No op for (I,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  int p0 = parameters[0][0];
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);
    
    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<var> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p1;
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iD) {
  SUCCEED() << "No op for (i,D) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_iv) {
  SUCCEED() << "No op for (i,v) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_id) {
  SUCCEED() << "No op for (i,d) input" << std::endl;
}

REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_id,
			   check_valid_iv,
			   check_invalid_id,
			   check_invalid_iv,
			   logprob_propto_id,
			   logprob_propto_iv,
			   gradient_finite_diff_id,
			   gradient_finite_diff_iv,
			   gradient_function_id,
			   gradient_function_iv,
			   vectorized_IV,
			   vectorized_ID,
			   vectorized_Iv,
			   vectorized_Id,
			   vectorized_iV,
			   vectorized_iD,
			   vectorized_iv,
			   vectorized_id);
#endif
