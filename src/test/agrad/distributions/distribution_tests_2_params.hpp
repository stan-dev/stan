#ifndef __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_PARAMS_HPP___
#define __TEST__AGRAD__DISTRIBUTIONS__DISTRIBUTION_TESTS_2_PARAMS_HPP___

// v: var
// d: double
// V: vector<var>
// D: vector<double>

using stan::agrad::var;

TYPED_TEST_P(AgradDistributionTestFixture, call_all_versions) {
  vector<double> parameters = this->first_valid_params();

  var param1, param2;
  var logprob;
  param1 = parameters[0];
  param2 = parameters[1];
  
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<true>(param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_<false>(param1, param2, errno_policy()));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2));
  EXPECT_NO_THROW(logprob = _LOG_PROB_(param1, param2, errno_policy()));
}

TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  params[1]))
      << "Failed with (d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(0.0, lp.val())
      << "Failed propto with (d,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_dv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(params[0],
					  var(params[1])))
      << "Failed with (d,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  params[1]))
      << "Failed with (v,d) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_valid_vv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params = parameters[n];
    var lp(0);
    EXPECT_NO_THROW(lp = _LOG_PROB_<true>(var(params[0]),
					  var(params[1])))
      << "Failed with (d,v) at index: " << n << std::endl
      << "(" << params[0] << ", " << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  invalid_params[1]),
		 std::domain_error)
      << "Default policy. "
      << "Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  invalid_params[1]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_dv) {  
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0],
				  var(invalid_params[1])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(invalid_params[0], 
				  var(invalid_params[1])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vd) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  invalid_params[1]),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  invalid_params[1]),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, check_invalid_vv) {
  vector<size_t> index;
  vector<double> invalid_values;

  const vector<double> valid_params = this->first_valid_params();
  TypeParam().invalid_values(index, invalid_values);
  ASSERT_EQ(index.size(), invalid_values.size());
  
  for (size_t n = 0; n < index.size(); n++) {
    vector<double> invalid_params(valid_params);
    invalid_params[index[n]] = invalid_values[n];

    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]),
				  var(invalid_params[1])),
		 std::domain_error)
      << "Default policy. Failed at index: " << n << std::endl
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
  for (size_t i = 0; i < valid_params.size(); i++) {
    vector<double> invalid_params(valid_params);
    invalid_params[i] = std::numeric_limits<double>::quiet_NaN();
    
    EXPECT_THROW(_LOG_PROB_<true>(var(invalid_params[0]), 
				  var(invalid_params[1])),
		 std::domain_error)
      << "Default policy with NaN for parameter: " << i
      << "(" << invalid_params[0] << "," << invalid_params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dd) {
  var logprob_true = _LOG_PROB_<true>(this->first_valid_params()[0],
				      this->first_valid_params()[1]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> params(parameters[n]);
    
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 params[1]);
    EXPECT_FLOAT_EQ(0.0,
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_dv) {
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(params[0],
					var(params[1]));
  var logprob_true = _LOG_PROB_<true>(params[0],
				      var(params[1]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(params[0],
					   var(params[1]));
    var logprob2_true = _LOG_PROB_<true>(params[0],
					 var(params[1]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vd) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					params[1]);
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      params[1]);
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   params[1]);
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 params[1]);
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, logprob_propto_vv) { 
  vector<double> params(this->first_valid_params());

  var logprob_false = _LOG_PROB_<false>(var(params[0]),
					var(params[1]));
  var logprob_true = _LOG_PROB_<true>(var(params[0]),
				      var(params[1]));
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    params[0] = parameters[n][0];
    params[1] = parameters[n][1];
    var logprob2_false = _LOG_PROB_<false>(var(params[0]),
					   var(params[1]));
    var logprob2_true = _LOG_PROB_<true>(var(params[0]),
					 var(params[1]));
    EXPECT_FLOAT_EQ((logprob_false - logprob2_false).val(), 
		    (logprob_true - logprob2_true).val())
      << "propto failed at index: " << n << std::endl
      << "_LOG_PROB_(" << this->first_valid_params()[0] << "," << this->first_valid_params()[1] << ") - " 
      << "_LOG_PROB_(" << params[0] << "," << params[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dd) {
  SUCCEED() << "No op for all double" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_dv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e) - _LOG_PROB_<false>(p[0], p[1]-e)) / e_times_2;
      
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
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1]) - _LOG_PROB_<false>(p[0]-e, p[1])) / e_times_2;
      
    var p0(p[0]);
    var lp = _LOG_PROB_<true>(p0, p[1]);
    vector<var> v_params(1);
    v_params[0] = p0;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_finite_diff_vv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  double e = this->e();
  double e_times_2 = (2.0 * e);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    double diff_g0 = (_LOG_PROB_<false>(p[0]+e, p[1]) - _LOG_PROB_<false>(p[0]-e, p[1])) / e_times_2;
    double diff_g1 = (_LOG_PROB_<false>(p[0], p[1]+e) - _LOG_PROB_<false>(p[0], p[1]-e)) / e_times_2;
      
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    vector<double> gradients;
    lp.grad(v_params, gradients);

    EXPECT_NEAR(diff_g0,
		gradients[0],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_NEAR(diff_g1,
		gradients[1],
		1e-4)
      << "Index: " << n << " - Finite diff test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dd) {
  SUCCEED() << "No op for (d,d,d) input" << std::endl;
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_dv) {  
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vd) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    
    var lp = _LOG_PROB_<true>(p0, p[1]);
    var expected_lp = TypeParam().log_prob(p0, p[1]);
    vector<var> v_params(1);
    v_params[0] = p0;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, gradient_function_vv) {  
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  for (size_t n = 0; n < parameters.size(); n++) {
    vector<double> p(parameters[n]);
    var p0(p[0]);
    var p1(p[1]);
    
    var lp = _LOG_PROB_<true>(p0, p1);
    var expected_lp = TypeParam().log_prob(p0, p1);
    vector<var> v_params(2);
    v_params[0] = p0;
    v_params[1] = p1;
    
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
      << "Index: " << n << " - hand-coded gradient test failed for parameter 0" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
    EXPECT_FLOAT_EQ(expected_gradients[1],
		    gradients[1])
      << "Index: " << n << " - hand-coded gradient test failed for parameter 1" << std::endl
      << "(" << p[0] << ", " << p[1] << ")" << std::endl;
  }
}



TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  vector<double> expected_grad_p0, expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1.push_back(grad[1]);
  }

  vector<var> p0, p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p0, grad_p1;
  logprob.grad(p0, grad_p0);
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_VD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
  }

  vector<var> p0;
  vector<double> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p0;
  logprob.grad(p0, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Vv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  double expected_grad_p1 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
    expected_grad_p1 += grad[1];
  }

  vector<var> p0;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p1);
  logprob.grad(p0, grad_p0);
  logprob.grad(x, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
  EXPECT_FLOAT_EQ(expected_grad_p1, grad_p1[0])
    << "Gradient failed for parameter 1"; 
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Vd) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  vector<double> expected_grad_p0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[n][0];
    double p1 = parameters[0][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0.push_back(grad[0]);
  }

  vector<var> p0;
  double p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p0;
  logprob.grad(p0, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p0[n], grad_p0[n]) 
      << "Index " << n << ": gradient failed for parameter 0"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_DV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  vector<double> p0;
  vector<var> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_Dv) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  double expected_grad_p1 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[n][0];
    var p1 = parameters[0][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1 += grad[0];
  }

  vector<double> p0;
  var p1 = parameters[0][1];
  for (size_t n = 0; n < parameters.size(); n++) {
    p0.push_back(parameters[n][0]);
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
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);

  double expected_logprob = 0.0;
  double expected_grad_p0 = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    var p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;

    x.push_back(p0);
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
    expected_grad_p1.push_back(grad[1]);
  }
  
  var p0 = parameters[0][0];
  vector<var> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p0, grad_p1;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);
  logprob.grad(p1, grad_p1);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
  for (size_t n = 0; n < parameters.size(); n++) {
    EXPECT_FLOAT_EQ(expected_grad_p1[n], grad_p1[n])
      << "Index " << n << ": gradient failed for parameter 1"; 
  }
}
TYPED_TEST_P(AgradDistributionTestFixture, vectorized_vD) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
    
  double expected_logprob = 0.0;
  double expected_grad_p0 = 0.0;
  for (size_t n = 0; n < parameters.size(); n++) {
    var p0 = parameters[0][0];
    double p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p0);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p0 += grad[0];
  }

  var p0 = parameters[0][0];
  vector<double> p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    p1.push_back(parameters[n][1]);
  }
  var logprob = _LOG_PROB_<true>(p0, p1);
  vector<double> grad_p0;
  vector<var> x;
  x.push_back(p0);
  logprob.grad(x, grad_p0);

  EXPECT_FLOAT_EQ(expected_logprob, logprob.val())
    << "log probability does not match" << std::endl;
  EXPECT_FLOAT_EQ(expected_grad_p0, grad_p0[0]) 
    << "Gradient failed for parameter 0"; 
}

TYPED_TEST_P(AgradDistributionTestFixture, vectorized_dV) {
  vector<vector<double> > parameters;
  TypeParam().valid_values(parameters);
  ASSERT_GT(parameters.size(), 0U);
  
  double expected_logprob = 0.0;
  vector<double> expected_grad_p1;
  for (size_t n = 0; n < parameters.size(); n++) {
    double p0 = parameters[0][0];
    var p1 = parameters[n][1];
    var logprob = _LOG_PROB_<true>(p0, p1);
    vector<var> x;
    x.push_back(p1);
    vector<double> grad;
    logprob.grad(x, grad);

    expected_logprob += logprob.val();
    expected_grad_p1.push_back(grad[0]);
  }

  double p0 = parameters[0][0];
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
// This has a limit of 50 tests.
REGISTER_TYPED_TEST_CASE_P(AgradDistributionTestFixture,
			   call_all_versions,
			   check_valid_dd,
			   check_valid_dv,
			   check_valid_vd,
			   check_valid_vv,
			   check_invalid_dd,
			   check_invalid_dv,
			   check_invalid_vd,
			   check_invalid_vv,
			   logprob_propto_dd,
			   logprob_propto_dv,
			   logprob_propto_vd,
			   logprob_propto_vv,
			   gradient_finite_diff_dd,
			   gradient_finite_diff_dv,
			   gradient_finite_diff_vd,
			   gradient_finite_diff_vv,
			   gradient_function_dd,
			   gradient_function_dv,
			   gradient_function_vd,
			   gradient_function_vv,
			   vectorized_VV,
			   vectorized_VD,
			   vectorized_Vv,
			   vectorized_Vd,
			   vectorized_DV,
			   vectorized_Dv,
			   vectorized_vV,
			   vectorized_vD,
			   vectorized_dV);
#endif
